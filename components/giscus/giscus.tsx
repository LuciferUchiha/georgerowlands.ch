import {useState, useEffect, ReactElement} from "react";
import Giscus from "@giscus/react";

export default function GiscusComments() {
    // This is done so that is loaded client side and skips hydration, no idea what this means but stackoverflow does
    const [showChild, setShowChild] = useState(false);
    const pathNameBlacklist = ["/", "/about"];
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
            id="giscus-comments"
            repo="LuciferUchiha/georgerowlands.ch"
            repoId="R_kgDOIt7MZQ"
            category="Announcements"
            categoryId="DIC_kwDOIt7MZc4CT8rz"
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
