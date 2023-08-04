import NextImage from "next/image";
import React from "react";

export default function Image({src, alt, width = 800, height= 0, caption = ""}) {

    return (
        <figure className="mt-8 flex flex-col items-center">
            <NextImage src={src} alt={alt} width={width} height={height}/>
            <figcaption className="mt-2 italic">{caption}</figcaption>
        </figure>
    );
};
