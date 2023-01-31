// Example from https://beta.reactjs.org/learn

import {useState} from 'react'

function MyButton() {
    const [count, setCount] = useState(0)

    function handleClick() {
        setCount(count + 1)
    }

    return (
        <div>
            <button onClick={handleClick}
                    className="bg-primary-700 hover:bg-primary-800 font-bold py-2 px-4 mt-8 border border-primary-700 rounded">
                Clicked {count} times
            </button>
        </div>
    )
}

export default function MyApp() {
    return <MyButton/>
}
