# Provenance report: {{ course }} / {{ chapter }}

{% if outcome.chapters -%}
Searched {{ outcome.chapters | join(", ") }}
{%- if outcome.auto_scoped %}, chosen automatically from the gaps in your notes{% endif %}.
{%- else -%}
No textbook chapter was searched.
{%- endif %}

Textbook tokens admitted: **{{ outcome.tokens_admitted }}** of a **{{ outcome.budget }}** budget.
Passages admitted: **{{ outcome.admitted | length }}**.
Rejected: **{{ outcome.rejections | length }}**.

# Rejections by mechanism

| Mechanism | Rejected |
|---|---|
{% for mechanism, count in by_mechanism -%}
| {{ mechanism }} | {{ count }} |
{% endfor %}
# Admitted passages

{% for passage in outcome.admitted -%}
- pages {{ passage.page_start }}-{{ passage.page_end }},
  necessity {{ '%.2f' % passage.necessity }}, {{ passage.token_estimate }} tokens
{% endfor %}
{% if not outcome.rejections %}
> Nothing was rejected. Spec 7.8: that is a bug report, not a success.
{% endif %}
