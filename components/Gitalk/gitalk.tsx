import {useState, useEffect} from "react";
import GitalkComponent from "gitalk/dist/gitalk-component";

export default function Gitalk() {
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
            <GitalkComponent options={{
                clientID: '680caa67f6ea23931525',
                clientSecret: '51b8ca0f000558106229c95bbe8cfb925943a7d9',
                repo: 'nextra-garden',
                owner: 'LuciferUchiha',
                distractionFreeMode: true,
            }}/>
        );
    }
}