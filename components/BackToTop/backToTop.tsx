import {useEffect} from "react";
import {VscChevronUp} from "react-icons/vsc";

export default function BackToTop() {

    const scrollToTop = () => {
        document.body.scrollTop = 0;
        document.documentElement.scrollTop = 0;
    };

    useEffect(() => {
        const onScroll = (event) => {
            // will exist because useEffect is after render cycle
            const backToTopButton = document.getElementById("backToTopButton");

            if (
                document.body.scrollTop > 20 ||
                document.documentElement.scrollTop > 20
            ) {
                backToTopButton.style.display = "block";
            } else {
                backToTopButton.style.display = "none";
            }
        };

        // after the component has rendered
        window.addEventListener("scroll", onScroll);
        // cleanup event handler when component is unmounted
        return () => {
            window.removeEventListener("scroll", onScroll);
        };
    }, []);

    return (<button
        id="backToTopButton"
        onClick={scrollToTop}
        className="fixed right-5 bottom-5 mt-8 p-2 hidden rounded-full bg-primary-700 border-primary-700 hover:bg-primary-800"
    >
        <VscChevronUp className="w-[36px] h-[36px] text-white"/>
    </button>);
}