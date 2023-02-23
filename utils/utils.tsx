export const randRange = (min: number, max: number) => Math.floor(Math.random() * (max - min + 1)) + min;

export const capitalizeFirstLetter = (str: string) => str.charAt(0).toUpperCase() + str.slice(1);
