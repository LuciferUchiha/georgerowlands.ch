import {useState} from "react";

export default function Counter() {
    const [count, setCount] = useState(0);

    function handleClick() {
        setCount(count + 1);
    }

    return (
            <button onClick={handleClick}
                    className="mt-8 rounded border px-4 py-2 font-bold text-white bg-primary-700 border-primary-700 hover:bg-primary-800">
                Clicked {count} times
            </button>
    );
}
