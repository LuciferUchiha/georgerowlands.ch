import { ReactTyped } from "react-typed";


const NameText = () => {
    return (
        <div>
            <ReactTyped 
                strings={["Hello there, "]}
                typeSpeed={40}
                showCursor={false}
            />
            <ReactTyped
                strings={["my name is"]}
                typeSpeed={40}
                startDelay={2000}
                showCursor={false}
            />

            <br />

            <ReactTyped
                strings={["George Rowlands"]}
                typeSpeed={40}
                startDelay={3000}
                showCursor={false}
            />
        </div>
    );
};

export default NameText;
