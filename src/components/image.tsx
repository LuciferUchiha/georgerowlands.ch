'use client';

import { Image as NextImage, type ImageProps as NextImageProps } from 'fumadocs-core/framework';
import { type ImgHTMLAttributes } from 'react';
import '../styles/image.css';
import Zoom, { type UncontrolledProps } from 'react-medium-image-zoom';

export type ImageProps = NextImageProps & {
  /**
   * Image props when zoom in
   */
  zoomInProps?: ImgHTMLAttributes<HTMLImageElement>;

  /**
   * Props for `react-medium-image-zoom`
   */
  rmiz?: UncontrolledProps;

  /**
   * Caption text to display below the image
   */
  caption?: string;

  /**
   * Width of the image
   */
  width?: number | string;

  /**
   * Height of the image
   */
  height?: number | string;
};

function getImageSrc(src: NextImageProps['src']): string {
  if (typeof src === 'string') return src;

  if (typeof src === 'object') {
    // Next.js
    if ('default' in src)
      return (src as { default: { src: string } }).default.src;
    return src.src;
  }

  return '';
}

export function Image({
  zoomInProps,
  children,
  rmiz,
  caption,
  alt,
  width = 800,
  height = 0,
  ...props
}: ImageProps) {
  // Use caption as alt text if caption not provided
  const imageAlt = caption || alt || '';

  const zoomContent = (
    <Zoom
      zoomMargin={20}
      wrapElement="span"
      {...rmiz}
      zoomImg={{
        src: getImageSrc(props.src),
        sizes: undefined,
        ...zoomInProps,
      }}
    >
      {children ?? (
        <NextImage
          width={width}
          height={height}
          alt={imageAlt}
          {...props}
        />
      )}
    </Zoom>
  );

  if (caption) {
    return (
      <figure className="flex flex-col items-center gap-2 my-6">
        {zoomContent}
        <figcaption className="text-sm text-gray-600 italic text-center max-w-prose">
          {caption}
        </figcaption>
      </figure>
    );
  }

  return zoomContent;
}
