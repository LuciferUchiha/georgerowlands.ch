import { Footer, Layout, Navbar } from 'nextra-theme-docs'
import { Head } from 'nextra/components'
import { getPageMap } from 'nextra/page-map'
import 'nextra-theme-docs/style.css'
import './globals.css'
import BuyMeACoffee from '../public/buymeacoffee.svg'
import Image from 'next/image'
 
export const metadata = {
  // Define your metadata here
  // For more information on metadata API, see: https://nextjs.org/docs/app/building-your-application/optimizing/metadata
}
 
const navbar = (
  <Navbar
    logo={<b>Nextra</b>}
    // ... Your additional navbar options
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
      >
        {/* Your additional tags should be passed as `children` of `<Head>` element */}
      </Head>
      <body>
        <Layout
          navbar={navbar}
          pageMap={await getPageMap()}
          sidebar={{ autoCollapse: true }}
          docsRepositoryBase="https://github.com/LuciferUchiha/georgerowlands.ch/tree/main/"
          footer={footer}
          // ... Your additional layout options
          nextThemes = {{
            defaultTheme: "dark",
          }}
          editLink= "Edit this page on GitHub ✍🏻"
          feedback={{
            content: "Questions or Suggestions? Give me feedback"
          }}
          toc={{
            title: "Table of Contents",
          }}
        >
          {children}
        </Layout>
      </body>
    </html>
  )
}