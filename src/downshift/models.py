"""The candidate model registry and the cost model behind the chart's x-axis.

The single most important decision in this file is that cost is computed from
**measured token counts**, never from a blended headline rate.

The usual "$/1M tokens, 50/50 input/output" figure is close to meaningless for
classification. A sentiment call is roughly 260 input tokens and 5-40 output
tokens -- about 87:1, nothing like 50:50 -- so a blend flatters expensive models
with cheap input and punishes cheap models with pricey output, and the ranking it
produces can invert against the actual bill. Worse, reasoning models bill their
`<think>` block as output tokens: measured locally, the same classification cost
260 output tokens with reasoning on and 37 with it off. A blended rate cannot see
that 7x at all.

So every row on the chart is priced as:

    cost_per_1k = 1000 * (mean_input_tokens * price_in + mean_output_tokens * price_out) / 1e6

with the token counts taken from the provider's own usage counters during the
same run that produced the accuracy number. Rate cards set the price; the run
sets the quantity.

PRICES ARE A SNAPSHOT (2026-09-09) and every entry carries its source. They move.
`verify_prices.py` (repo root) re-checks every Azure-metered rate against the
live catalogue; the rest -- Anthropic-direct and Fireworks-direct -- are not in
that catalogue and must be re-read by hand before anything is published.

This file previously named a `verify_prices_live` helper that was never written,
so nothing had ever been re-checked. deepseek-v4-flash's cached rate had drifted
to $0.007, a figure matching no meter at any tier or scope, on a row sitting on
the published Pareto frontier.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# Where the model runs, which decides both how we call it and how we price it.
LOCAL, HOSTED = "local", "hosted"

# What role the model plays in the story the chart tells.
FRONTIER = "frontier"    # the bill you are trying to cut
CHEAP = "cheap"          # proprietary small tier
OPEN = "open"            # open-weight, someone else serves it
ENCODER = "encoder"      # fine-tuned classifier, not a generative model
TRIVIAL = "trivial"      # tf-idf, majority class: the honesty floor


# A price of UNKNOWN means "we could not verify this", not "free". Rows carrying
# it are still evaluated for accuracy but are excluded from the cost axis --
# plotting an unverified price is how a chart starts lying.
UNKNOWN = -1.0


@dataclass
class ModelSpec:
    key: str                       # our short name, used in results and on the chart
    runtime: str                   # LOCAL or HOSTED
    tier: str
    label: str                     # display name for the chart
    call_id: str = ""              # litellm model string, or ollama tag for LOCAL
    price_in: float = 0.0          # USD per 1M input tokens
    price_out: float = 0.0         # USD per 1M output tokens
    price_cached: float = -1.0     # USD per 1M *cached* input tokens; -1 = fall back to price_in
    price_cache_write: float = -1.0 # USD per 1M tokens to CREATE a cache entry; -1 = derive.
                                    # Anthropic charges a 1.25x premium on the write and we
                                    # were billing it as ordinary input, so every run
                                    # understated its own cost. Derived as 1.25x price_in for
                                    # any model that publishes a cached-read rate (i.e. has a
                                    # cache tier at all), else price_in.
    cache_min_tokens: int = 0      # provider's minimum cacheable prompt length; 0 = unknown/none
    # Limits of the DEPLOYMENT, not of the model. Both read off the Foundry
    # portal's catalogue rows on 2026-09-10 (the "Context window" / "Token
    # limits" fields), which is the only place they are published for a given
    # deployment -- they are not in the retail price API, so verify_prices.py
    # cannot check them and they must be re-read by hand when a model is added.
    #
    # These routinely disagree with what the model's maker publishes, and the
    # deployment's figure is the one that binds at call time:
    #
    #   Mistral Large 3   vendor 256k ctx, no output cap   Azure 128k / 4,096
    #   Grok 4.1 Fast     vendor 2M ctx                    Azure 128k  (16x)
    #   GPT-4.1-mini      OpenAI 1,047,576                 Azure 1,048,576
    #
    # The vendor's own numbers live in model_cards.json, and a gap between the
    # two must be written into that card's `caveat` -- tests_invariants.py
    # enforces it, so an undocumented divergence fails the suite rather than
    # quietly becoming folklore.
    #
    # 0 means NOT RECORDED, never "no limit" and never "zero". The Fireworks
    # rows sit at 0 because this catalogue does not cover them.
    context_window: int = 0        # DEPLOYMENT's context window, tokens; 0 = unknown
    max_output: int = 0            # DEPLOYMENT's max output tokens; 0 = unknown
    source: str = ""               # where the price came from
    env_key: str = ""              # env var that must be set to use it
    api_base_suffix: str = ""      # path appended to AZURE_API_BASE, resolved lazily
    drops_system_role: bool = False # HOSTED: endpoint silently discards the system
                                    # message (accepts it, 200s, throws it away). Folds
                                    # it into the user turn -- see system_role_lm.py.
    think: bool | None = None      # LOCAL (ollama) only: force reasoning off/on
    extra_body: dict | None = None # HOSTED only: passed through litellm's extra_body,
                                    # e.g. {"chat_template_kwargs": {"enable_thinking": False}}
                                    # for hosts (vLLM/SGLang-backed, e.g. Fireworks) that honor it.
                                    # Verified NOT honored by ollama's own /api/chat -- that path
                                    # needs OllamaLM's native `think` field instead (see ollama_lm.py).
    notes: str = ""

    @property
    def available(self) -> bool:
        if self.runtime == LOCAL:
            return True
        if not (self.env_key and os.environ.get(self.env_key)):
            return False
        # A key without an endpoint is not usable; say so here rather than
        # letting the call fail later as a confusing auth error.
        return bool(self.api_base) if self.api_base_suffix else True

    @property
    def api_base(self) -> str:
        """Endpoint, resolved when the model is called rather than at import.

        Resolving this eagerly in the module body reads the environment before
        .env has necessarily been loaded, yielding an empty api_base. litellm
        then silently falls back to the provider's public endpoint and sends an
        Azure key to api.openai.com, which fails as an auth error and looks like
        a bad key rather than a bad URL. Lazy resolution makes import order
        irrelevant.
        """
        if not self.api_base_suffix:
            return ""
        host = os.environ.get("AZURE_API_BASE", "").rstrip("/")
        return f"{host}{self.api_base_suffix}" if host else ""

    def cache_applies(self, mean_in_tokens: float) -> tuple[bool, str]:
        """Can this model cache a prompt of this size, and if not, why not?

        Providers publish a MINIMUM cacheable prompt length and silently skip
        caching below it -- no error, no warning, the tokens just bill at full
        price. Claude Haiku 4.5's minimum is 4,096 tokens while Sonnet 4.6's is
        1,024, so on a 2,640-token prompt Sonnet caches and Haiku cannot. That
        single threshold collapses their real cost gap from 3x to 1.2x, which
        is invisible if you only read the rate card.

        An unrecorded minimum is NOT a verified zero. It reads as a pass with the
        uncertainty named, exactly as `fits` treats an unrecorded limit -- the two
        used to disagree, and this one said "verified fine" where the other said
        "nobody checked". That mattered: 8 hosted rows publish a cached rate at
        0.10-0.20x input with no minimum on file, so a silent pass here applies a
        5-10x discount on a threshold no one has confirmed, and it errs in the
        flattering direction.
        """
        if not self.cache_min_tokens:
            return True, (f"{self.label}: cache minimum not on file, not checked -- "
                          f"a discounted rate here is assumed, not verified")
        if mean_in_tokens >= self.cache_min_tokens:
            return True, ""
        return False, (f"prompt is {mean_in_tokens:.0f} tokens; {self.label} needs "
                       f"{self.cache_min_tokens:,}+ to cache, so every token bills at full rate")

    def fits(self, max_output_needed: int, prompt_tokens: int = 0) -> tuple[bool, str]:
        """Can this deployment serve a task needing this much output and context?

        Returns (ok, reason). `reason` is empty when ok and unqualified.

        A limit of 0 means NOT STATED, never "no limit" and never "zero". An
        unknown limit cannot disqualify a model -- that would silently drop
        every Fireworks row, whose limits the Azure catalogue does not carry --
        so it passes with the uncertainty named in `reason` instead. Being
        explicit about which of the two kinds of pass you got is the point:
        "verified to fit" and "nobody checked" must not read the same.
        """
        if self.max_output and max_output_needed > self.max_output:
            return False, (f"needs {max_output_needed:,} output tokens; "
                           f"{self.label} caps at {self.max_output:,}")
        need_ctx = prompt_tokens + max_output_needed
        if self.context_window and prompt_tokens and need_ctx > self.context_window:
            return False, (f"needs {need_ctx:,} tokens of context "
                           f"({prompt_tokens:,} prompt + {max_output_needed:,} output); "
                           f"{self.label} caps at {self.context_window:,}")
        unknown = [n for n, v in (("output limit", self.max_output),
                                  ("context window", self.context_window)) if not v]
        if unknown:
            return True, f"{self.label}: {' and '.join(unknown)} not on file, not checked"
        return True, ""

    @property
    def card(self):
        """The vendor's own description, or None if nobody has transcribed it.

        Kept in `model_cards` rather than as a field here: the text runs to
        paragraphs, and inlining fifteen of them would bury the price table
        this file exists to be.
        """
        from .model_cards import CARDS
        return CARDS.get(self.key)

    @property
    def price_known(self) -> bool:
        return self.price_in >= 0 and self.price_out >= 0

    @property
    def cache_write_rate(self) -> float:
        """Per-1M rate for tokens that CREATE a cache entry.

        Anthropic bills a cache write at 1.25x the base input rate, and only
        subsequent reads at the discounted rate. We recorded reads and threw the
        write quantity away, so writes were charged at 1.0x -- the 0.25x premium
        went unbilled. Small but real and it moved in the flattering direction:
        measured on banking77, 12 of 251 calls wrote a 1,884-token prefix
        (num_threads=12 requests are all in flight before any reply lands, so
        every one of them misses), understating Sonnet by 1.96% and Opus 1.99%.
        """
        if self.price_cache_write >= 0:
            return self.price_cache_write
        return self.price_in * 1.25 if self.price_cached >= 0 else self.price_in

    def cost_per_1k_calls(self, mean_in_tokens: float, mean_out_tokens: float,
                          mean_cached_tokens: float = 0.0,
                          mean_cache_write_tokens: float = 0.0) -> float:
        """Dollars per 1,000 classifications at this row's measured token usage.

        Cached input is billed separately and much cheaper (typically 0.1x, and
        every provider here publishes a cached rate). Ignoring it is not a
        rounding error on this workload: a 77-class prompt carries a ~1,900-token
        label list that is byte-identical on every call, and providers cache it
        automatically -- measured, 1,792 of 1,943 tokens (92%) came back cached
        from the second call onward. Charging all of that at the full input rate
        overstates a many-class run by roughly 6x.

        Returns NaN when the rate card could not be verified, so the caller has
        to decide what to do rather than silently charting a made-up zero.
        """
        if not self.price_known:
            return float("nan")
        cached = max(0.0, min(mean_cached_tokens, mean_in_tokens))
        # Writes are part of prompt_tokens too (litellm folds them in), so they
        # come out of the same budget as `fresh` rather than adding to it --
        # otherwise the input would be double-counted. What changes is the RATE
        # they are charged at: 1.25x base, not 1.0x.
        writes = max(0.0, min(mean_cache_write_tokens, mean_in_tokens - cached))
        fresh = mean_in_tokens - cached - writes
        cached_rate = self.price_cached if self.price_cached >= 0 else self.price_in
        per_call = (fresh * self.price_in
                    + cached * cached_rate
                    + writes * self.cache_write_rate
                    + mean_out_tokens * self.price_out) / 1e6
        return per_call * 1000


# ── local candidates (verified pulled and callable on this machine) ─────────
#
# Priced against the published rate for a comparable hosted open-weight model,
# because $0 marginal cost on a laptop is true but useless for a buying decision.
# Every local row is a *proxy* for what that model would cost on a real endpoint,
# and the chart must say so.

LOCAL_MODELS = [
    ModelSpec("qwen3-0.6b-direct", LOCAL, OPEN, "Qwen3 0.6B (direct)",
              call_id="qwen3:0.6b", think=False,
              price_in=0.01, price_out=0.05,
              source="deepinfra sub-1B tier (contested - see notes)",
              notes="Cheapest tier price is disputed between sources; treat as indicative."),
    ModelSpec("qwen3-1.7b-direct", LOCAL, OPEN, "Qwen3 1.7B (direct)",
              call_id="qwen3:1.7b", think=False,
              price_in=0.03, price_out=0.12,
              source="https://www.together.ai/pricing (small-model tier)"),
    ModelSpec("qwen3-1.7b-reasoning", LOCAL, OPEN, "Qwen3 1.7B (reasoning on)",
              call_id="qwen3:1.7b", think=True,
              price_in=0.03, price_out=0.12,
              source="https://www.together.ai/pricing (small-model tier)",
              notes="Same weights and same rate card as the row above. Any cost "
                    "difference between them is purely reasoning tokens."),
    ModelSpec("qwen3-4b-direct", LOCAL, OPEN, "Qwen3 4B (direct)",
              call_id="qwen3:4b", think=False,
              price_in=0.10, price_out=0.15,
              source="https://deepinfra.com/Qwen/Qwen3.5-9B (nearest served size)"),
    ModelSpec("gemma3-4b-direct", LOCAL, OPEN, "Gemma 3 4B (direct)",
              call_id="gemma3:latest", think=None,
              price_in=0.07, price_out=0.34,
              source="https://deepinfra.com/models/text-generation (gemma tier)"),
    ModelSpec("gpt-oss-20b-direct", LOCAL, OPEN, "gpt-oss-20B (direct)",
              call_id="gpt-oss:20b", think=False,
              price_in=0.03, price_out=0.14,
              source="https://deepinfra.com/openai/gpt-oss-20b"),
]

# ── hosted candidates ──────────────────────────────────────────────────────
#
# Prices verified 2026-09-09 against each provider's official page. `call_id`
# strings are NOT yet verified against a live API -- validate_call_ids() checks
# them once credentials exist, because a stale model id is the most likely way
# this run dies halfway through.

# OpenAI-compatible path: serves the OpenAI-family deployments.
AZURE_OPENAI_SUFFIX = "/openai/v1"

# Anthropic-native path. Claude deployments on this resource are NOT reachable
# over the OpenAI-compatible route -- both /openai/v1/chat/completions and
# /models/chat/completions return 404 api_not_supported for them, while
# OpenAI-family models on the same resource work fine. Verified working:
# POST <host>/anthropic/v1/messages with an `x-api-key` header, which litellm
# reaches via the `anthropic/` model prefix plus this api_base.
AZURE_ANTHROPIC_SUFFIX = "/anthropic"

# Shared provenance for the 2026-09-10 batch. Spelled out because the tier and
# scope qualifiers are load-bearing: the same model carries Flex, Standard,
# Priority and Batch meters at up to 2x spread, and Global/DataZone/Regional
# differ by ~10% on top of that. A price without them is not a price.
_RETAIL_2026_09_10 = ("https://prices.azure.com/api/retail/prices "
                      "(Foundry Models, Standard + Global scope, verified 2026-09-10)")


# ── hosted candidates: Azure AI Foundry deployments ───────────────────────
#
# All nine verified reachable on 2026-09-09. Prices are filled in from the
# pricing research and carry their source; any left as UNKNOWN are evaluated
# for accuracy but kept off the cost axis.

HOSTED_MODELS = [
    # -- OpenAI family, via the OpenAI-compatible endpoint --
    ModelSpec("gpt-5.4-nano", HOSTED, CHEAP, "GPT-5.4-nano",
              call_id="openai/gpt-5.4-nano", env_key="AZURE_API_KEY",
              api_base_suffix=AZURE_OPENAI_SUFFIX,
              context_window=400_000, max_output=128_000,
              price_in=0.20, price_out=1.25,
              price_cached=0.02,
              cache_min_tokens=1024,
              source="https://prices.azure.com/api/retail/prices (Global Standard)",
              notes="Measured 0 reasoning tokens on a one-word classification."),
    ModelSpec("gpt-5-nano", HOSTED, CHEAP, "GPT-5-nano",
              call_id="openai/gpt-5-nano", env_key="AZURE_API_KEY",
              api_base_suffix=AZURE_OPENAI_SUFFIX,
              context_window=400_000, max_output=128_000,
              price_in=0.05, price_out=0.40,
              price_cached=0.005,
              cache_min_tokens=1024,
              source="https://prices.azure.com/api/retail/prices (Global Standard)",
              notes="Measured 64 reasoning tokens for a one-word answer -- billed as output."),
    ModelSpec("gpt-5.4-mini", HOSTED, CHEAP, "GPT-5.4-mini",
              call_id="openai/gpt-5.4-mini-2", env_key="AZURE_API_KEY",
              api_base_suffix=AZURE_OPENAI_SUFFIX,
              context_window=400_000, max_output=128_000,
              price_in=0.75, price_out=4.50,
              price_cached=0.075,
              cache_min_tokens=1024,
              source="https://prices.azure.com/api/retail/prices (Global Standard)",
              notes="Deployment name carries a '-2' suffix; underlying model is gpt-5.4-mini."),
    ModelSpec("gpt-4.1-mini", HOSTED, CHEAP, "GPT-4.1-mini",
              call_id="openai/gpt-4.1-mini", env_key="AZURE_API_KEY",
              api_base_suffix=AZURE_OPENAI_SUFFIX,
              context_window=1_000_000, max_output=32_768,
              price_in=0.40, price_out=1.60,
              price_cached=0.1,
              cache_min_tokens=1024,
              source="https://prices.azure.com/api/retail/prices (Global Standard)"),
    ModelSpec("gpt-5.2", HOSTED, FRONTIER, "GPT-5.2",
              call_id="openai/gpt-5.2", env_key="AZURE_API_KEY",
              api_base_suffix=AZURE_OPENAI_SUFFIX,
              context_window=400_000, max_output=128_000,
              price_in=1.75, price_out=14.00,
              price_cached=0.175,
              cache_min_tokens=1024,
              source="https://prices.azure.com/api/retail/prices (Global Standard)"),

    # -- third-party open-ish models on the same endpoint --
    ModelSpec("deepseek-v4-flash", HOSTED, OPEN, "DeepSeek-V4-Flash",
              call_id="openai/DeepSeek-V4-Flash", env_key="AZURE_API_KEY",
              api_base_suffix=AZURE_OPENAI_SUFFIX,
              context_window=1_000_000, max_output=128_000,
              price_in=0.15, price_out=0.31,
              price_cached=0.030,
              source="https://prices.azure.com/api/retail/prices (Data Zone; meters "
                     "'FW Deepseek-v4-Flash In/Opt/Cd In DZ' -- the FIREWORKS-served "
                     "listing inside the Foundry catalogue, verified 2026-09-10)",
              notes="Two different meter families sell these same weights on Azure and they "
                    "are not interchangeable: the Fireworks listing used here is $0.15/$0.31 "
                    "with cached at $0.030, while the native 'Azure Deepseek Models' listing "
                    "is $0.19/$0.51 with cached at $0.028 (Global). Cheaper on input and "
                    "output, dearer on cached reads. Which one bills a given deployment "
                    "depends on which catalogue entry it was created from -- worth confirming "
                    "against an actual invoice before publishing this row. The cached rate "
                    "was stored as $0.007 until verify_prices.py flagged that it matched no "
                    "meter at any tier or scope; it is 4.3x that."),
    ModelSpec("phi-4-mini-instruct", HOSTED, OPEN, "Phi-4-mini-instruct",
              call_id="openai/Phi-4-mini-instruct", env_key="AZURE_API_KEY",
              api_base_suffix=AZURE_OPENAI_SUFFIX,
              context_window=128_000, max_output=4_096,
              price_in=0.075, price_out=0.30,
              drops_system_role=True,
              source="https://prices.azure.com/api/retail/prices "
                     "(Foundry Models / Azure Phi Models; meters 'Phi-4-Mini-Input Tokens' "
                     "and 'Phi-4-Mini-Output Tokens', one flat rate across 13 regions)",
              notes="NO cached-input meter exists for any Phi model, so every input token "
                    "bills at full rate. That is the whole story on a many-class task: the "
                    "77-label prompt is ~1,900 byte-identical tokens per call, and the models "
                    "that do cache get ~92% of it back at a tenth of the price. On paper "
                    "$0.075/1M undercuts GPT-5-nano's $0.05; in practice it lands above "
                    "DeepSeek-V4-Flash. 3.8B dense, MIT-licensed -- see model_cards.py."),
    ModelSpec("kimi-k2.6", HOSTED, OPEN, "Kimi K2.6",
              call_id="openai/Kimi-K2.6", env_key="AZURE_API_KEY",
              api_base_suffix=AZURE_OPENAI_SUFFIX,
              context_window=262_144, max_output=262_144,
              price_in=0.95, price_out=4.00,
              price_cached=0.16,
              source="https://prices.azure.com/api/retail/prices (Global Standard; meter literally named 'K2.6 Thinking')",
              notes="Emitted 126 output tokens for a one-word answer -- thinking is on by default."),

    # -- Anthropic family, via the Anthropic-native path --
    ModelSpec("claude-sonnet-4-6", HOSTED, FRONTIER, "Claude Sonnet 4.6",
              call_id="anthropic/claude-sonnet-4-6", env_key="AZURE_API_KEY",
              api_base_suffix=AZURE_ANTHROPIC_SUFFIX,
              context_window=1_000_000, max_output=128_000,
              price_in=3.00, price_out=15.00,
              price_cached=0.3,
              cache_min_tokens=1024,
              source="https://platform.claude.com/docs/en/about-claude/pricing "
                     "(CCU billing on Foundry converts these exact per-token rates 1:1)",
              notes="Sampling params are deprecated on recent Claude models -- do not send temperature."),
    ModelSpec("claude-haiku-4-5", HOSTED, CHEAP, "Claude Haiku 4.5",
              call_id="anthropic/claude-haiku-4-5", env_key="AZURE_API_KEY",
              api_base_suffix=AZURE_ANTHROPIC_SUFFIX,
              context_window=200_000, max_output=64_000,
              price_in=1.00, price_out=5.00, price_cached=0.10,
              cache_min_tokens=4096,
              source="https://platform.claude.com/docs/en/about-claude/pricing "
                     "(cache hits 0.1x base; CCU billing on Foundry is these same rates)",
              notes="Anthropic's cheap tier -- 3x under Sonnet 4.6 on both axes."),
    # Claude Opus 4.8 is retired from the default candidate set: at $5/$25 it
    # cost more than the rest of the board combined and never won a task.
    # Kept registered (not deleted) so an explicit run can still reference it.
    ModelSpec("claude-opus-4-8", HOSTED, FRONTIER, "Claude Opus 4.8",
              call_id="anthropic/claude-opus-4-8", env_key="AZURE_API_KEY",
              api_base_suffix=AZURE_ANTHROPIC_SUFFIX,
              context_window=1_000_000, max_output=128_000,
              price_in=5.00, price_out=25.00,
              price_cached=0.5,
              cache_min_tokens=1024,
              source="https://platform.claude.com/docs/en/about-claude/pricing "
                     "(CCU billing on Foundry converts these exact per-token rates 1:1)"),

    # ── candidates added 2026-09-10 from the Foundry serverless catalogue ──
    #
    # Selected under one hard constraint: nothing above the Haiku tier
    # ($1.00/$5.00). Every rate below is the STANDARD, GLOBAL meter. That
    # qualifier matters more than it looks -- the same model is metered at three
    # service tiers, and reading the wrong one off the API misprices it by up to
    # 2x. For gpt-5.6-luna: Flex $0.10/$0.60, Standard $0.20/$1.20, Priority
    # $0.40/$2.40. Flex is genuinely half price but is a deferred-execution tier,
    # so it is the wrong choice for a latency-measured benchmark. Batch is
    # cheaper again (50% off) and wrong for the same reason.
    #
    # `price_cached` is left at -1 (= bill cached input at the full input rate)
    # wherever no cached meter EXISTS on the catalogue. That is not laziness: on
    # the 77-class task the prompt is ~1,900 byte-identical tokens per call, and
    # a model with no cached tier pays full price on all of them. Measured on
    # phi-4-mini-instruct, that alone is a 2x cost penalty relative to a model
    # with the same headline rate that does cache. Four of these ten have no
    # cached meter, and it will show up on the chart rather than in the rate card.
    #
    # None of these are deployed yet -- `available` will read True as soon as
    # AZURE_API_KEY is set, but calls 404 until the deployment exists. The
    # deployment name must match `call_id` exactly.

    # -- xAI. A vendor we had nothing from. The fast tier prices reasoning and
    # non-reasoning IDENTICALLY, which is a controlled experiment we cannot buy
    # anywhere else: same weights, same rate, one switch. Our DIRECT/REASONING
    # profiles measure the prompt-side of that; this measures the weights side.
    ModelSpec("grok-4-1-fast-non-reasoning", HOSTED, CHEAP, "Grok 4.1 Fast (non-reasoning)",
              call_id="openai/grok-4-1-fast-non-reasoning", env_key="AZURE_API_KEY",
              api_base_suffix=AZURE_OPENAI_SUFFIX,
              context_window=128_000, max_output=128_000,
              price_in=0.20, price_out=0.50,
              source=_RETAIL_2026_09_10 + "; meters 'Grok 4.1 Inp/Outp Glbl'",
              notes="No cached-input meter published for any Grok model, so the label "
                    "list bills at full rate on every call. 128k output limit."),
    ModelSpec("grok-4-1-fast-reasoning", HOSTED, CHEAP, "Grok 4.1 Fast (reasoning)",
              call_id="openai/grok-4-1-fast-reasoning", env_key="AZURE_API_KEY",
              api_base_suffix=AZURE_OPENAI_SUFFIX,
              context_window=128_000, max_output=128_000,
              price_in=0.20, price_out=0.50,
              source=_RETAIL_2026_09_10 + "; meters 'Grok 4.1 Inp/Outp Glbl'",
              notes="Same rate card as the non-reasoning deployment, so any cost gap "
                    "between the two rows is pure output-token volume."),

    # -- OpenAI tiers we were missing. gpt-5-mini is the notable one: 3x under
    # the gpt-5.4-mini already on the board, on both axes.
    ModelSpec("gpt-5-mini", HOSTED, CHEAP, "GPT-5-mini",
              call_id="openai/gpt-5-mini", env_key="AZURE_API_KEY",
              api_base_suffix=AZURE_OPENAI_SUFFIX,
              context_window=400_000, max_output=128_000,
              price_in=0.25, price_out=2.00, price_cached=0.025,
              cache_min_tokens=1024,
              source=_RETAIL_2026_09_10 + "; meters 'GPT 5 Mini Inpt/outpt/cchd Inpt Glbl'",
              notes="Standard tier. The 'pp' (Priority Processing) meters are 1.8x this "
                    "($0.45/$3.60) and Batch is half ($0.125/$1.00, 24h turnaround)."),
    ModelSpec("gpt-4.1-nano", HOSTED, CHEAP, "GPT-4.1-nano",
              call_id="openai/gpt-4.1-nano", env_key="AZURE_API_KEY",
              api_base_suffix=AZURE_OPENAI_SUFFIX,
              context_window=1_000_000, max_output=32_768,
              price_in=0.10, price_out=0.40, price_cached=0.025,
              cache_min_tokens=1024,
              source=_RETAIL_2026_09_10 + "; meters 'gpt 4.1 nano Inp/Outp/cached Inp glbl'",
              notes="Catalogue-flagged Legacy, and CHEAPEST of the 2026-09-10 batch per "
                    "classification (~$0.109/1k on the 77-class prompt) despite four others "
                    "having a lower headline output rate -- because it has a cached-input "
                    "meter at $0.025 and they do not. Judged on the rate card alone it looks "
                    "like the weakest of the ten; judged on the workload it is the best. "
                    "Note gpt-5-nano is still half its input rate ($0.05) but is "
                    "reasoning-tuned and measured 391 output tokens for a one-word answer, "
                    "which is what actually decides the bill."),
    ModelSpec("gpt-5.6-luna", HOSTED, CHEAP, "GPT-5.6-luna",
              call_id="openai/gpt-5.6-luna", env_key="AZURE_API_KEY",
              api_base_suffix=AZURE_OPENAI_SUFFIX,
              context_window=1_100_000, max_output=128_000,
              price_in=0.20, price_out=1.20, price_cached=0.02,
              cache_min_tokens=1024,
              source=_RETAIL_2026_09_10 + "; meters '5.6 luna ShortCo Inp/Opt/Cd Inp Std Gl'",
              notes="SHORT-CONTEXT rate. GPT-5.6 meters context-tier separately and the "
                    "long-context rate is 2x ($0.40/$1.80); our prompts are ~2k tokens so "
                    "the short tier applies, but a long-document task would reprice. "
                    "Cheapest of the three 5.6 variants by a wide margin -- terra is "
                    "$2.00/$12.00 and sol $5.00/$30.00, both above the Haiku ceiling."),

    # -- vendors we had nothing from at any price.
    ModelSpec("mistral-large-3", HOSTED, CHEAP, "Mistral Large 3",
              call_id="openai/Mistral-Large-3", env_key="AZURE_API_KEY",
              api_base_suffix=AZURE_OPENAI_SUFFIX,
              context_window=128_000, max_output=4_096,
              price_in=0.50, price_out=1.50,
              source=_RETAIL_2026_09_10 + "; meters 'Large 3 Inp/Outp glbl'",
              notes="A flagship priced under most cheap tiers -- its $1.50 output undercuts "
                    "Haiku's $5.00 by 3.3x. No cached meter. 4.1k output limit, which is "
                    "ample for a label but would bind on a generative task."),
    ModelSpec("cohere-command-a-plus", HOSTED, CHEAP, "Cohere Command A Plus",
              call_id="openai/Cohere-command-a-plus-05-2026", env_key="AZURE_API_KEY",
              api_base_suffix=AZURE_OPENAI_SUFFIX,
              context_window=128_000, max_output=64_000,
              price_in=0.80, price_out=3.20,
              source=_RETAIL_2026_09_10 + "; meters 'Command A Plus Inp/Outp Glbl 1M'",
              notes="Cohere, a vendor with no representation on the board. Note the plain "
                    "'Command A' meter is 3x this ($2.50/$10.00) -- Plus is the cheaper "
                    "product despite the name. No cached meter. 64k output limit."),

    # -- open-weight models one generation back from what we already run. This is
    # the product's own thesis applied to the board: the cheaper sibling may
    # clear the bar, and nobody checks.
    ModelSpec("deepseek-v3.2", HOSTED, OPEN, "DeepSeek-V3.2",
              call_id="openai/DeepSeek-V3.2", env_key="AZURE_API_KEY",
              api_base_suffix=AZURE_OPENAI_SUFFIX,
              context_window=128_000, max_output=128_000,
              price_in=0.58, price_out=1.68,
              source=_RETAIL_2026_09_10 + "; meters 'V3.2 Inp/Outp glbl'",
              notes="No cached meter, unlike the V4 generation which has one -- so V3.2 is "
                    "cheaper on the rate card than V4-Flash on output but may lose on a "
                    "many-class prompt once caching is accounted for. Exactly the "
                    "comparison the cost axis exists to settle."),
    ModelSpec("deepseek-v3.2-speciale", HOSTED, OPEN, "DeepSeek-V3.2-Speciale",
              call_id="openai/DeepSeek-V3.2-Speciale", env_key="AZURE_API_KEY",
              api_base_suffix=AZURE_OPENAI_SUFFIX,
              context_window=128_000, max_output=128_000,
              price_in=0.58, price_out=1.68,
              source=_RETAIL_2026_09_10 + "; meters 'V3.2 SP Inp/Outp glbl'",
              notes="Metered identically to plain V3.2, so the two rows differ only in "
                    "weights. Weakest justification of the ten -- include it only if the "
                    "V3.2 row proves interesting."),
    ModelSpec("kimi-k2.5", HOSTED, OPEN, "Kimi K2.5",
              call_id="openai/Kimi-K2.5", env_key="AZURE_API_KEY",
              api_base_suffix=AZURE_OPENAI_SUFFIX,
              context_window=262_144, max_output=262_144,
              price_in=0.60, price_out=3.00, price_cached=0.10,
              source=_RETAIL_2026_09_10 + "; meters 'K2.5 Thinking Inp/Outp glbl', 'K2.5 cached glbl'",
              notes="The cheaper sibling of the kimi-k2.6 already on the board "
                    "($0.95/$4.00/$0.16) -- 37% less input, 25% less output. K2.6 emitted "
                    "126 output tokens for a one-word answer because thinking is on by "
                    "default; expect the same here and budget output accordingly."),

    # -- Fireworks serverless: a price-diverse slice, not the whole catalogue.
    # Picked to span the cheap end where a cost-cut story actually lives -- from
    # $0.05/1M (cheapest MoE Fireworks sells) up through a mid-tier dense/MoE
    # comparison point -- rather than by name recognition or benchmark buzz.
    # Prices verified against https://docs.fireworks.ai/serverless/pricing
    # (Standard tier) on 2026-09-09.
    ModelSpec("nemotron-lightning-30b", HOSTED, OPEN, "Nemotron 3.5 Lightning 30B-A3B",
              call_id="fireworks_ai/accounts/fireworks/models/nemotron-lightning-3p5-30b-a3b",
              env_key="FIREWORKS_API_KEY",
              price_in=0.05, price_out=0.20,
              price_cached=0.01,
              source="https://docs.fireworks.ai/serverless/pricing",
              extra_body={"chat_template_kwargs": {"enable_thinking": False}},
              notes="Cheapest priced model in the Fireworks catalogue on either axis. "
                    "Reasoning is on by default (measured 328 output tokens on a one-word "
                    "answer); the chat_template_kwargs override is verified working on "
                    "Fireworks (328 tok -> 2 tok, 2.6s -> 0.2s, same correct answer) -- "
                    "confirmed NOT reachable the same way on ollama, so this is host-specific."),
    ModelSpec("glm-5.3-flash", HOSTED, OPEN, "GLM 5.3 Flash",
              call_id="fireworks_ai/accounts/fireworks/models/glm-5p3-flash",
              env_key="FIREWORKS_API_KEY",
              price_in=0.15, price_out=0.50,
              price_cached=0.03,
              source="https://docs.fireworks.ai/serverless/pricing"),
    ModelSpec("kimi-k2.6-fireworks", HOSTED, OPEN, "Kimi K2.6 (Fireworks)",
              call_id="fireworks_ai/accounts/fireworks/models/kimi-k2p6",
              env_key="FIREWORKS_API_KEY",
              price_in=0.95, price_out=4.00,
              price_cached=0.16,
              source="https://docs.fireworks.ai/serverless/pricing",
              notes="Fireworks serverless counterpart of the Azure kimi-k2.6 row. Added 2026-09-11 as the "
                    "mid-tier Phase 2 incumbent after qwen3.7-plus turned out not to be deployed "
                    "(listed by GET /v1/models, 404 on completions). Price row verified on the "
                    "pricing page that day: $0.95 / $0.16 cached / $4.00 per 1M."),
    ModelSpec("deepseek-v4-pro-fireworks", HOSTED, OPEN, "DeepSeek V4 Pro (Fireworks)",
              call_id="fireworks_ai/accounts/fireworks/models/deepseek-v4-pro-0813",
              env_key="FIREWORKS_API_KEY",
              price_in=1.32, price_out=3.96,
              price_cached=0.044,
              source="https://docs.fireworks.ai/serverless/pricing",
              notes="DeepSeek's flagship on Fireworks serverless (dated 0813 snapshot, for reproducibility). "
                    "Added 2026-09-11 as the mid-tier Phase 2 incumbent replacing kimi-k2.6, which cost 15x "
                    "the cheap tier per item (637 reasoning tokens per one-word answer) for lower accuracy. "
                    "Price row verified on the pricing page that day: $1.32 / $0.044 cached / $3.96 per 1M."),
    ModelSpec("minimax-m3", HOSTED, OPEN, "MiniMax M3",
              call_id="fireworks_ai/accounts/fireworks/models/minimax-m3",
              env_key="FIREWORKS_API_KEY",
              price_in=0.30, price_out=1.20,
              price_cached=0.06,
              source="https://docs.fireworks.ai/serverless/pricing"),
    ModelSpec("qwen3.7-plus", HOSTED, OPEN, "Qwen 3.7 Plus",
              call_id="fireworks_ai/accounts/fireworks/models/qwen3p7-plus",
              env_key="FIREWORKS_API_KEY",
              price_in=0.40, price_out=1.60,
              price_cached=0.08,
              source="https://docs.fireworks.ai/serverless/pricing",
              notes="Same blended price bracket as gpt-4.1-mini on Azure -- a direct comparison point."),
    ModelSpec("gpt-oss-120b-fireworks", HOSTED, OPEN, "gpt-oss-120B (Fireworks)",
              call_id="fireworks_ai/accounts/fireworks/models/gpt-oss-120b",
              env_key="FIREWORKS_API_KEY",
              price_in=0.15, price_out=0.60,
              price_cached=0.015,
              source="https://docs.fireworks.ai/serverless/pricing",
              notes="Hosted 120B counterpart to the local gpt-oss:20b row -- same family, larger, priced."),
]

# ── non-generative baselines ───────────────────────────────────────────────
#
# These exist so the chart cannot lie by omission. If a TF-IDF model trained in
# four seconds matches a prompted LLM, that is the finding, and a chart that
# left it out would have sold an optimizer nobody needed.

BASELINE_MODELS = [
    ModelSpec("majority", LOCAL, TRIVIAL, "Always predict majority class",
              price_in=0.0, price_out=0.0,
              notes="The floor. Any model below this line is worse than a constant."),
    ModelSpec("tfidf-logreg", LOCAL, TRIVIAL, "TF-IDF + logistic regression",
              price_in=0.0, price_out=0.0,
              source="CPU-only; cost rounds to zero at any realistic volume."),
    ModelSpec("modernbert", LOCAL, ENCODER, "ModernBERT-base (fine-tuned)",
              price_in=0.0, price_out=0.0,
              source="priced separately from measured CPU throughput"),
]


def suitable_for(specs, max_output_needed: int, prompt_tokens: int = 0):
    """Split candidates into those that can serve a task and those that cannot.

    Returns (ok, rejected, unverified) where `rejected` and `unverified` are
    lists of (spec, reason). Exists so a sweep can drop an impossible model
    *before* spending on it: a task needing 8k of output has no business
    calling a deployment capped at 4,096, and finding that out from a
    mid-sweep truncation costs both money and a corrupted row.

    `unverified` is kept separate from `ok` on purpose -- those passed only
    because nobody has recorded their limits, and a caller publishing a
    "checked" claim needs to know the difference.
    """
    ok, rejected, unverified = [], [], []
    for spec in specs:
        fine, why = spec.fits(max_output_needed, prompt_tokens)
        if not fine:
            rejected.append((spec, why))
        elif why:
            unverified.append((spec, why))
            ok.append(spec)
        else:
            ok.append(spec)
    return ok, rejected, unverified

ALL_MODELS = LOCAL_MODELS + HOSTED_MODELS + BASELINE_MODELS
BY_KEY = {m.key: m for m in ALL_MODELS}


def available_models(include_baselines: bool = True) -> list[ModelSpec]:
    """Every candidate we can actually call right now.

    Hosted rows silently drop out when their key is absent, so a partial .env
    yields a smaller chart rather than a crashed run.
    """
    pool = ALL_MODELS if include_baselines else LOCAL_MODELS + HOSTED_MODELS
    return [m for m in pool if m.available]


def missing_keys() -> dict[str, list[str]]:
    """Which env vars would unlock which additional models. Reported, not hidden."""
    out: dict[str, list[str]] = {}
    for m in HOSTED_MODELS:
        if not m.available:
            out.setdefault(m.env_key, []).append(m.label)
    return out
