import React from 'react';

const Test = ({ message } : { message: string }) => {
    return (
        <div>
            <h1 className="text-3xl font-bold underline"
            >{message}</h1>
            <p>This component renders a message inside an h1 tag.</p>
        </div>
    );
};

export default Test;