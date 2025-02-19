{% extends "index.md.j2" %}

{%- block mathblock -%}
```math
{{ cell.source | strip_dollars }}
```
{%- endblock mathblock -%}

## Add a space after rawcell
{% block rawcell %}
{{ super() }}

{% endblock rawcell %}

{%- block output_group -%}
{%- if cell.metadata.html_output  %}
    {{ super() }}
{%- else  -%}
<div className="code-output-wrapper">
```
{{ super().strip('\n') }}
```
</div>
{%- endif %}
{% endblock output_group %}