{% extends "index.md.j2" %}

{%- block output_group -%}
{%- if not cell.metadata.html_output  %}
<br/>
{{ super()}}
{%- endif %}
{% endblock output_group %}