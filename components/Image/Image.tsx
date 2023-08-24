import NextImage from "next/image";
import Caption from "../Caption/Caption";

export default function Image({src, alt, width = 800, height = 0, caption = ""}) {
    return (
        <Caption caption={caption}>
            <NextImage src={src} alt={alt} width={width} height={height}/>
        </Caption>
    );
};
