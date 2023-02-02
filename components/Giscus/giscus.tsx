import {useState, useEffect} from "react";

export default function Giscus() {
    const [showChild, setShowChild] = useState(false);
    useEffect(() => {
        setShowChild(true);
    }, []);

    if (!showChild) {
        return null;
    }

    if (typeof window === 'undefined') {
        return <></>;
    } else {
        return (
            <script src="https://giscus.app/client.js"
                    data-repo="LuciferUchiha/nextra-garden"
                    data-repo-id="R_kgDOIt7MZQ"
                    data-category="Announcements"
                    data-category-id="DIC_kwDOIt7MZc4CT8rz"
                    data-mapping="pathname"
                    data-strict="0"
                    data-reactions-enabled="1"
                    data-emit-metadata="0"
                    data-input-position="top"
                    data-theme="transparent_dark"
                    data-lang="en"
                    crossOrigin="anonymous"
                    async>
            </script>
        );
    }
}