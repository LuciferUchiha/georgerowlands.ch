import {useState, useEffect} from "react";
import VisGraph from "react-graph-vis";

export default function Graph({graph, options, events}) {
    const [showChild, setShowChild] = useState(false);
    useEffect(() => {
        setShowChild(true);
    }, []);

    if (!showChild) {
        return null;
    }

    if (typeof window === "undefined") {
        return <></>;
    } else {
        const defaultOptions = {
            edges: {
                color: "#000000"
            },
            height: "500px"
        };
        // merge the default options with incoming options
        options = {...defaultOptions, ...options};

        return (
            <div className="border-2 border-primary-600 rounded-md dark:border-primary-700">
            <VisGraph
                graph={graph}
                options={options}
                events={events}
            />
            </div>
        );
    }
}