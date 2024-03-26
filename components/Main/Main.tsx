import { Inter } from "next/font/google";
// to use google fonts
const inter = Inter({ subsets: ["latin"] });

export default function Main({ children }) {
  return <main className={inter.className}>{children}</main>;
}