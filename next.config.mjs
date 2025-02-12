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

export default withNextra({
    images: {
        formats: [ 'image/webp' ],
        // it can 0 also for disabling
        minimumCacheTTL: 60
    },
});