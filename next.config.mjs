import nextra from 'nextra';
import simplePlantUML from '@akebifiky/remark-simple-plantuml';

const withNextra = nextra({
    theme: 'nextra-theme-docs',
    themeConfig: './theme.config.tsx',
    staticImage: true,
    latex: true,
    flexsearch: {
        codeblocks: false
    },
    defaultShowCopyCode: true,
    mdxOptions: {
        remarkPlugins: [
            [
                simplePlantUML,
                {baseUrl: "https://www.plantuml.com/plantuml/svg"}
            ]
        ]
    }
});

export default withNextra();