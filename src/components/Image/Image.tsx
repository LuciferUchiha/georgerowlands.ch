import NextImage from "next/image";
import Caption from "@components/Caption/Caption";

export default function Image({
  src,
  alt = "",
  width = 800,
  height = 0,
  caption = "",
}: {
  src: string;
  alt?: string;
  width?: number;
  height?: number;
  caption?: string;
}) {
  // if a caption is defined but not an alt text, use the caption as alt text
  if (!alt && caption) {
    alt = caption;
  }

  return (
    <div className="flex flex-col items-center">
      <Caption caption={caption}>
        <NextImage src={src} alt={alt} width={width} height={height} />
      </Caption>
    </div>
  );
}
