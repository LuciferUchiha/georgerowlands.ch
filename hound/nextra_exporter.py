import os
import os.path

from nbconvert.exporters.markdown import MarkdownExporter


# -----------------------------------------------------------------------------
# Classes
# -----------------------------------------------------------------------------

class NextraExporter(MarkdownExporter):
    export_from_notebook = "Nextra Markdown"

    @property
    def template_paths(self):
        """
        We want to inherit from HTML template, and have template under
        ``./templates/`` so append it to the search path. (see next section)

        Note: nbconvert 6.0 changed ``template_path`` to ``template_paths``
        """
        template_paths = super()._template_paths() + [os.path.join(os.path.dirname(__file__), "templates")]
        #print(template_paths)
        return template_paths

    def _template_file_default(self):
        """
        We want to use the new template we ship with our library.
        """
        return 'nextra_template.tpl'  # full
