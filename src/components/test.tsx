import React from 'react';

const Test = ({ message } : { message: string }) => {
    return (
        <div>
            <h1>{message}</h1>
            <p>This component renders a message inside an h1 tag.</p>
        </div>
    );
};

export default Test;