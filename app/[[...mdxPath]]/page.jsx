import { generateStaticParamsFor, importPage } from 'nextra/pages'
import { useMDXComponents } from '~/src/mdx-components'
import GiscusComments from '@components/GiscusComments/GiscusComments'

export const generateStaticParams = generateStaticParamsFor('mdxPath')

export async function generateMetadata(props) {
  const params = await props.params
  const { metadata } = await importPage(params.mdxPath)

  // when we are at the root params is an empty object
  const title = Object.keys(params).length === 0 ? 'georgerowlands.ch' : `${metadata.title} - GR`;

  return {
    ...metadata,
    title,
  }
}

const Wrapper = useMDXComponents().wrapper

export default async function Page(props) {
  const params = await props.params
  const result = await importPage(params.mdxPath)
  const { default: MDXContent, toc, metadata } = result
  return (
    <Wrapper toc={toc} metadata={metadata}>
      <MDXContent {...props} params={params} />
      <div className="mt-8">
        <GiscusComments />
      </div>
    </Wrapper>
  )
}