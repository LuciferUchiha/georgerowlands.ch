import nextra from 'nextra'
import simplePlantUML from '@akebifiky/remark-simple-plantuml';
 
const withNextra = nextra({
  // ... Other Nextra config options
  latex: {
    renderer: 'mathjax',
    options: {
      config: {
        loader: {load: ['[tex]/color']},
        tex: {
          packages: {'[+]': ['color']},
          macros: {
            // natural, integer, rational, real, complex numbers
            NN: '\\mathbb{N}',
            ZZ: '\\mathbb{Z}',
            QQ: '\\mathbb{Q}',
            RR: '\\mathbb{R}',
            CC: '\\mathbb{C}',
            // probability
            P: '\\mathbb{P}',
            F: '\\mathcal{F}', // sigma-algebra
            E: '\\mathbb{E}', // expectation
            V: '\\mathbb{V}', // variance
          }
        }
      }
    }
  },
  search: {
    codeblocks: false,
  },
  defaultShowCopyCode: true,
  mdxOptions: {
    rehypePrettyCodeOptions: {
      theme: {
        dark: "one-dark-pro",
        light: "one-light"
      }
    },
    remarkPlugins: [
      [
          simplePlantUML,
          {baseUrl: "https://www.plantuml.com/plantuml/svg"}
      ]
    ]
  }
})
 
// You can include other Next.js configuration options here, in addition to Nextra settings:
export default withNextra({
  // ... Other Next.js config options
  reactStrictMode: true,
  // ignore .ipynb files
  webpack: (config) => {
    config.module.rules.push({
      test: /\.ipynb$/,
      use: 'ignore-loader'
    })
    return config
  }
})
