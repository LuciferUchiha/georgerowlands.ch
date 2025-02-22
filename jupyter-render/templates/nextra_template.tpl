{% extends "index.md.j2" %}


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