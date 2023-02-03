import {useState} from "react";

export default function Counter() {
    const [count, setCount] = useState(0);

    function handleClick() {
        setCount(count + 1);
    }

    return (
            <button onClick={handleClick}
                    className="bg-primary-700 hover:bg-primary-800 font-bold text-white py-2 px-4 mt-8 border border-primary-700 rounded">
                Clicked {count} times
            </button>
    );
}
