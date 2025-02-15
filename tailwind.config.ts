import type { Config } from 'tailwindcss'

const config: Config = {
  content: [
    './app/**/*.{js,jsx,ts,tsx}',
    './content/**/*.{md,mdx}',
    './src/**/*.{js,jsx,ts,tsx}'
  ],
  theme: {
    extend: {}
  },
  plugins: []
}

export default config