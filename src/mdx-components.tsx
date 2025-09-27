import defaultMdxComponents from 'fumadocs-ui/mdx';
import type { MDXComponents } from 'mdx/types';
import { Callout } from '@/components/callout';
import { Image } from '@/components/image';

// use this function to get MDX components, you will need it for rendering MDX
export function getMDXComponents(components?: MDXComponents): MDXComponents {
  return {
    ...defaultMdxComponents,
    Callout,
    Image,
    // eslint-disable-next-line jsx-a11y/alt-text, @typescript-eslint/no-explicit-any
    img: (props) => <Image {...(props as any)} />,
    ...components,
  };
}
