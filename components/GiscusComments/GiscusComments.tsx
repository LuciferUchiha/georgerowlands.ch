import {useState, useEffect} from "react";
import { useRouter } from 'next/router';
import Giscus from "@giscus/react";

export default function GiscusComments() {
    // ensure giscus is reloaded when client side route is changed
    const { route } = useRouter();
    // This is done so that is loaded client side and skips hydration, no idea what this means but stackoverflow does
    const [showChild, setShowChild] = useState(false);
    const pathNameBlacklist = ["/"];
    useEffect(() => {
        setShowChild(true);
    }, []);

    if (!showChild) {
        return null;
    }

    if (typeof window === "undefined") {
        return <></>;
    } else {

        const comments = pathNameBlacklist.indexOf(window.location.pathname) > -1 ? <></> : <Giscus
            key={route}
            id="giscus-comments"
            repo="LuciferUchiha/georgerowlands.ch"
            repoId="R_kgDOIt7MZQ"
            category="Digital Garden Giscus"
            categoryId="DIC_kwDOIt7MZc4ChZB7"
            mapping="pathname"
            reactionsEnabled="1"
            emitMetadata="0"
            inputPosition="top"
            theme="transparent_dark"
            lang="en"
            loading="lazy"
        />;
        return comments;
    }
}