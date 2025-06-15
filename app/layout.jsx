import { Footer, Layout, Navbar } from 'nextra-theme-docs'
import { Head, Search } from 'nextra/components'
import { getPageMap } from 'nextra/page-map'
import 'nextra-theme-docs/style.css'
import './globals.css'
import BuyMeACoffee from '../public/buyMeACoffee.svg'
import Image from 'next/image'
import { Analytics } from '@vercel/analytics/next';
 
export const metadata = {
  // Define your metadata here
  // For more information on metadata API, see: https://nextjs.org/docs/app/building-your-application/optimizing/metadata
  description: "George Rowlands' Digital Garden. A place to learn about Computer Science, Mathematics, Machine Learning and Artificial Intelligence. Study notes for students by a student.",
  metadataBase: new URL('https://georgerowlands.ch'),
  keywords: [
    'Nextra',
    'Next.js',
    'React',
    'JavaScript',
    'MDX',
    'Markdown',
    'Static Site Generator',
    'georgerowlands.ch',
    'George Rowlands',
    'Digital Garden',
    'Study Notes',
    'Computer Science',
    'Software Engineering',
    'Machine Learning',
    'Data Science',
    'Mathematics'
  ],
  generator: 'Next.js',
  applicationName: 'georgerowlands.ch',
  appleWebApp: {
    title: 'georgerowlands.ch',
  },
}

const logo = (
  <>
    <svg
      version="1.0"
      xmlns="http://www.w3.org/2000/svg"
      width="36px"
      height="36px"
      viewBox="0 0 128 128"
      preserveAspectRatio="xMidYMid meet"
    >
      <g transform="translate(0,128) scale(0.1,-0.1)">
        <path
          fill="currentColor"
          d="M707 1214 c-187 -45 -463 -255 -529 -401 -69 -154 143 -163 395 -17
                                          44 25 81 43 83 41 6 -5 -54 -41 -141 -85 -134 -67 -344 -223 -400 -296 -15
                                          -19 -29 -53 -32 -74 -5 -38 -4 -40 32 -52 133 -44 298 71 483 337 110 157 112
                                          158 282 158 106 0 119 -2 168 -28 51 -27 86 -67 98 -115 19 -72 -46 -161 -159
                                          -217 -81 -40 -221 -78 -268 -73 l-31 3 53 80 c28 44 76 111 105 149 30 37 54
                                          74 54 82 0 53 -93 -45 -202 -213 l-73 -112 -41 -1 c-25 0 -47 -6 -54 -15 -11
                                          -13 -9 -20 13 -43 15 -16 27 -32 27 -36 0 -5 -21 -51 -47 -102 -26 -51 -45
                                          -96 -41 -99 17 -18 49 13 92 90 26 47 49 85 51 85 2 0 61 -31 132 -69 150 -81
                                          176 -91 171 -65 -2 11 -53 45 -133 88 -169 92 -156 82 -143 111 8 17 22 26 47
                                          30 227 36 342 78 427 157 52 48 74 90 74 142 0 145 -198 254 -394 217 -33 -6
                                          -61 -9 -63 -8 -2 2 13 25 32 50 19 25 35 48 35 51 0 21 -57 -3 -128 -54 -193
                                          -138 -406 -222 -450 -178 -38 38 7 117 135 238 134 127 313 222 423 224 49 1
                                          55 -1 58 -20 2 -12 -9 -35 -24 -53 -18 -22 -24 -35 -16 -43 23 -23 92 47 92
                                          95 0 55 -73 70 -193 41z m-156 -536 c-147 -213 -272 -320 -375 -322 -34 -1
                                          -41 2 -44 19 -7 53 141 187 340 308 61 37 115 67 120 67 5 0 -13 -33 -41 -72z"
        />
      </g>
    </svg>
    <span className="ml-4 font-bold">George Rowlands</span>
  </>
)
 
const navbar = (
  <Navbar
    logo={logo}
    // ... Your additional navbar options
    projectLink="https://github.com/LuciferUchiha/georgerowlands.ch/tree/main/"
  />
)

const search = (
  <Search
    emptyResult="Couldn't find anything in the garden."
    loading="Searching the garden..."
    placeholder="Search within the garden..."
  />
)

const footer = (
<Footer>
  <div className="flex w-full flex-row items-center justify-between">
    <ul>
      <li>
        <p className="font-bold">Socials</p>
      </li>
      <li>
        <a
          href="https://www.linkedin.com/in/georgerowlands/"
          className="text-sm"
        >
          Linkedin↗
        </a>
      </li>
      <li>
        <a href="https://github.com/LuciferUchiha" className="text-sm">
          GitHub↗
        </a>
      </li>
      <li>
        <a
          href="https://stackoverflow.com/users/10994912/george-r"
          className="text-sm"
        >
          StackOverflow↗
        </a>
      </li>
    </ul>
    <a href="https://www.buymeacoffee.com/georgerowlands" target="_blank">
      <Image
        src={BuyMeACoffee}
        alt="Buy Me A Coffee"
        width={250}
      />
    </a>
  </div>
</Footer>
)

export default async function RootLayout({ children }) {

  return (
    <html
      // Not required, but good for SEO
      lang="en"
      // Required to be set
      dir="ltr"
      // Suggested by `next-themes` package https://github.com/pacocoursey/next-themes#with-app
      suppressHydrationWarning
    >
      <Head
      // ... Your additional head options
        color={{
          hue: 270
        }}
      >
        {/* Your additional tags should be passed as `children` of `<Head>` element */}
        <link rel="icon" type="image/x-icon" href="/logo.svg"/>
      </Head>
      <body>
        <Layout
          navbar={navbar}
          search={search}
          pageMap={await getPageMap()}
          sidebar={{ autoCollapse: true, defaultMenuCollapseLevel: 1 }}
          docsRepositoryBase="https://github.com/LuciferUchiha/georgerowlands.ch/tree/main/"
          footer={footer}
          // ... Your additional layout options
          nextThemes = {{
            defaultTheme: "light",
            attribute: 'class',
          }}
          editLink= "Edit this page on GitHub ✍🏻"
          feedback={{
            content: "Questions or Suggestions?"
          }}
          toc={{
            title: "Table of Contents",
          }}
        >
          {children}
        </Layout>
        <Analytics />
      </body>
    </html>
  )
}