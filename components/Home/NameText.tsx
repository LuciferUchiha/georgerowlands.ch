import { ReactTyped } from "react-typed";


const NameText = () => {
    return (
        <div>
            <ReactTyped 
                strings={["Hello there,"]}
                className="text-3xl"
                typeSpeed={40}
                showCursor={false}
            />
            <ReactTyped
                strings={["my name is"]}
                className="text-3xl ml-2"
                typeSpeed={40}
                startDelay={2000}
                showCursor={false}
            />

            <br />

            <ReactTyped
                strings={["George Rowlands"]}
                className="font-bold text-primary-500 dark:text-primary-700 text-6xl"
                typeSpeed={40}
                startDelay={3000}
                showCursor={false}
            />
        </div>
    );
};

export default NameText;
