import {useState, useEffect} from "react";
import VisGraph from "react-graph-vis";

import 'vis-network/styles/vis-network.css';

export default function Graph({graph, options, events, directed = false, nodeSize = 25, edgeWidth = 2}) {
    const [showChild, setShowChild] = useState(false);
    useEffect(() => {
        setShowChild(true);
    }, []);

    if (!showChild) {
        return null;
    }

    function formatSingleLetterNodes(nodes: any) {
        return nodes.map((node: any) => {
            if (node.label.length === 1) {
                return {...node, ...{label: " " + node.label + " "}};
            } else {
                return node;
            }
        });
    }

    if (typeof window === "undefined") {
        return <></>;
    } else {

        const defaultOptions = {
            edges: {
                width: edgeWidth,
                color: "#7300E6",
                font: {
                    color: "#FFFFFF",
                    size: "12",
                    strokeWidth: 0,
                }
            },
            nodes: {
                color: "#7300E6",
                font: {
                    color: "#FFFFFF",
                },
                scaling: {
                    label: {
                        enabled: true,
                        min: 12,
                        max: 30
                    },
                    min: 1,
                    max: 100
                },
                shape: "circle",
            },
            height: "500px"
        };
        // merge the default options with incoming options
        options = {...defaultOptions, ...options};

        graph.nodes = formatSingleLetterNodes(graph.nodes);

        if (!directed) {
            options = {...options, ...{edges: {...options.edges, ...{arrows: {to: false, from: false}}}}}
        }
        return (
            <div className="mt-5 border-2 border-primary-600 rounded-md dark:border-primary-700">
                <VisGraph
                    graph={graph}
                    options={options}
                    events={events}
                />
            </div>
        );
    }
}