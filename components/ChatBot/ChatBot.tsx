import { BsRobot } from "react-icons/bs";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "~/components/ui/sheet";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "~/components/ui/select";
import { Input } from "~/components/ui/input";
import { Label } from "~/components/ui/label";
import { Textarea } from "~components/ui/textarea";
import { Button } from "~components/ui/button";
import axios from "axios";
import { useState, useEffect } from "react";
import Markdown from "react-markdown";
import rehypeKatex from "rehype-katex";
import remarkMath from "remark-math";
import "katex/dist/katex.min.css";
import remarkGfm from "remark-gfm";

type Role = "system" | "user" | "assistant";

type Message = {
  role: Role;
  content: string;
};

type Context = {
  id: string;
  page_content: string;
  path: string;
  title: string;
  score: number;
};

const MessageCard = ({ role, content }: Message) => {
  return (
    <div
      className={`flex flex-row ${
        role === "assistant" ? "justify-start" : "justify-end"
      }`}
    >
      <div
        className={`px-4 max-w-[90%] rounded-lg ${
          role === "assistant"
            ? "bg-zinc-800 text-zinc-50"
            : "bg-zinc-100 text-zinc-800"
        }`}
      >
        <div id="markdown">
          <Markdown
            remarkPlugins={[remarkMath, remarkGfm]}
            rehypePlugins={[rehypeKatex]}
            components={{
              p: ({ children, ...props }) => (
                <p className="py-4" {...props}>
                  {children}
                </p>
              ),
              pre: ({ children, ...props }) => (
                <pre
                  className={`overflow-x-auto ${
                    role === "assistant" ? "bg-zinc-900" : "bg-zinc-200"
                  } p-4 rounded-md mb-4`}
                  {...props}
                >
                  {children}
                </pre>
              ),
              li: ({ children, ...props }) => (
                <li className="list-disc ml-12" {...props}>
                  {children}
                </li>
              ),
              code: ({ children, ...props }) => (
                <code
                  className={`overflow-x-auto italic font-normal ${
                    role === "assistant" ? "bg-zinc-900" : "bg-zinc-200"
                  } p-1 rounded-md`}
                  {...props}
                >
                  {children}
                </code>
              ),
              a: ({ children, ...props }) => {
                const isEternalLink = (props as any).href.startsWith("http");
                const textColor = isEternalLink
                  ? "text-blue-500"
                  : "text-primary-400";

                return (
                  <a
                    className={`${textColor} underline font-semibold`}
                    {...props}
                  >
                    {children}
                  </a>
                );
              },
            }}
          >
            {content}
          </Markdown>
        </div>
      </div>
    </div>
  );
};

const MessageList = ({ messages }: { messages: Message[] }) => {
  // scrollable div
  return (
    <div
      className="flex flex-col overflow-y-auto gap-4 flex-1 h-[62vh] md:h-[65vh]"
      id="message-list"
    >
      {messages.map((message, index) => (
        <MessageCard
          key={index}
          role={message.role}
          content={message.content}
        />
      ))}
    </div>
  );
};

const scrollToBottom = (id: string) => {
  const scrollableElement = document.getElementById(id);
  if (scrollableElement) {
    setTimeout(() => {
      scrollableElement.scrollTo(0, scrollableElement.scrollHeight);
    }, 0);
  }
};

const isJsonString = (str: string) => {
  try {
      JSON.parse(str);
  } catch (e) {
      return false;
  }
  return true;
}

const getPrompt = (question: string, contexts: Context[]) => {
  const parsedContexts = contexts.map((context) => {
    return `[${context.title}](${context.path}):\n${context.page_content}\n\n`;
  });
  const context = parsedContexts.join("");

  return `Please answer the following question, using the following documents. \
The link of each document is written before the content. If \
any of the documents where helpful, include their links after \
the sentence in the answer that they were helpful for. Write the link exactly as provided \
do not change or add any text. \
\
If the question is just general conversation or chit-chat do not use or include any \
references in the answer. \
\
Write any math equations identically as they appear in the documents, \
between single or double dollar signs. \
\
At the end of the answer, ask the user if they have any further questions. \
\
Documents: \
${context}
\
Question: \
${question}
\
Answer:`;
};

export default function ChatBot() {
  const [threshold, setThreshold] = useState<number>(0.4);
  const [apiKey, setApiKey] = useState<string>("");
  const [model, setModel] = useState<string>("gpt-4o");
  const [inputValue, setInputValue] = useState<string>("");
  const [messages, setMessages] = useState<Message[]>([]);

  useEffect(() => {
    const value = localStorage.getItem("apiKey") || "";
    setApiKey(value);
  }, []);

  const saveKeyToLocalStorage = (key: string) => {
    localStorage.setItem("apiKey", key);
  };

  const addMessage = (role: Role, content: string) => {
    setMessages((messages) => [...messages, { role, content }]);
  };

  const updateLastMessage = (content: string) => {
    setMessages((messages) => {
      const newMessages = [...messages];
      newMessages[messages.length - 1].content += content;
      return newMessages;
    });
  };

  const handleMessageSend = () => {
    const input = inputValue.trim();
    if (input.length === 0) return;

    addMessage("user", input);
    rag(input);
    setInputValue("");
    scrollToBottom("message-list");
  };

  const rag = async (question: string) => {
    try {
      // Step 1: Embed the question
      const embedResponse = await axios.post(
        "https://api.openai.com/v1/embeddings",
        {
          input: question,
          model: "text-embedding-3-small",
        },
        {
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${apiKey}`,
          },
        }
      );
      const embedding = embedResponse.data.data[0].embedding;
      console.log("embedding", embedding);

      // Step 2: Get context using the embedding
      const contextResponse = await axios.post(
        "https://digital-garden-br88je8.svc.aped-4627-b74a.pinecone.io/query",
        {
          vector: embedding,
          topK: 8,
          includeMetadata: true,
        },
        {
          headers: {
            "Api-Key": process.env.NEXT_PUBLIC_PINECONE_KEY || "",
          },
        }
      );
      const matches = contextResponse.data.matches;
      const contexts = matches.map(
        (match: { id: string; score: number; metadata: any }) => {
          const title = match.metadata["title"];
          const cleanPath = match.metadata.full_path
            .replaceAll("../pages", "")
            .replaceAll(".mdx", "");
          return {
            id: match.id,
            page_content: match.metadata.page_content,
            path: cleanPath,
            title: title,
            score: match.score,
          };
        }
      );
      console.log("contexts", contexts);
      const filteredContexts = contexts.filter(
        (context) => context.score > threshold
      );
      console.log("filteredContexts", filteredContexts);

      // Step 3: Send the message along with the context
      const prompt = getPrompt(question, filteredContexts);
      const conversation = [
        {
          role: "system",
          content: "You are a helpful assistant.",
        },
        ...messages,
        {
          role: "user",
          content: prompt,
        },
      ]
      console.log("conversation", conversation);
      const messageResponse = await fetch(
        "https://api.openai.com/v1/chat/completions",
        {
          method: "POST",
          headers: {
            Authorization: `Bearer ${apiKey}`,
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            model: model,
            messages: conversation,
            stream: true,
          }),
        }
      );
      const stream = messageResponse.body?.getReader();

      if (!stream) {
        throw new Error("No stream");
      }

      addMessage("assistant", "");

      while (true) {
        const { done, value } = await stream.read();
        if (done) {
          break;
        }

        const text = new TextDecoder().decode(value);
        if (text) {
          const chunks = text.split("\n").filter((chunk) => chunk.length > 0);
          for (let chunk of chunks) {
            chunk = chunk.replace("data: ", "");
            if (isJsonString(chunk)) {
              const messageChunk = JSON.parse(chunk);
              const deltaContent = messageChunk.choices[0].delta.content;
              if (deltaContent) {
                updateLastMessage(messageChunk.choices[0].delta.content);
                scrollToBottom("message-list");
              }
            }
          }
        }
      }
    } catch (error) {
      console.error(error);
      addMessage(
        "assistant",
        "Sorry, there was an error processing your request."
      );
    }
    console.log(messages);
  };

  return (
    <Sheet>
      <SheetTrigger>
        <BsRobot className="w-[24px] h-[24px]" />
      </SheetTrigger>
      <SheetContent
        side="right"
        className="dark:border-zinc-800 min-w-full md:min-w-[800px] min-h-screen flex flex-col"
      >
        <SheetHeader>
          <SheetTitle>Talk to the Digital Garden</SheetTitle>
          <SheetDescription className="hidden md:block">
            Using Retrieval Augmented Generation (RAG) you can talk to the
            garden and ask questions about the content. The assistant will try
            to help you find the answer and provide references to the content.
          </SheetDescription>
        </SheetHeader>
        <div className="mt-4">
          <div className="flex flex-row justify-between flex-wrap gap-4 flex-none">
            <div className="flex flex-col w-96">
              <div className="flex flex-row">
                <Label htmlFor="apiKey" className="mb-2 mr-2">
                  OpenAI API Key
                </Label>
              </div>
              <Input
                id="apiKey"
                placeholder="sk-********"
                value={apiKey}
                onChange={(e) => {
                  setApiKey(e.target.value);
                  saveKeyToLocalStorage(e.target.value);
                }}
              />
            </div>
            <div className="flex flex-col w-32">
              <Label className="mb-2">Threshold (K=8)</Label>
              <Input
                type="number"
                defaultValue={threshold}
                value={threshold}
                onChange={(e) => {
                  const value = parseFloat(e.target.value);
                  if (value < 0 || value > 1) return;
                  setThreshold(value);
                }}
              />
            </div>
            <div className="flex flex-col w-48">
              <Label className="mb-2">Model</Label>
              <Select
                defaultValue="gpt-4o"
                onValueChange={(value) => setModel(value)}
              >
                <SelectTrigger>
                  <SelectValue placeholder="gpt-4o" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="gpt-4o">GPT-4o</SelectItem>
                  <SelectItem value="gpt-4o-mini">GPT-4o Mini</SelectItem>
                  <SelectItem value="gpt-4">GPT-4</SelectItem>
                  <SelectItem value="gpt-4-turbo">GPT-4 Turbo</SelectItem>
                  <SelectItem value="gpt-3.5-turbo">GPT-3.5 Turbo</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>
          <div className="mt-4">
            <MessageList messages={messages} />
          </div>
          <div className="my-6 gap-4 w-full flex flex-row flex-none">
            <Textarea
              className="resize-none text-base w-[640px] bg-zinc-200 dark:bg-zinc-800"
              placeholder="Send a message..."
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyDown={(e) => {
                // if it is shift+enter, work as a new line
                if (e.key === "Enter" && e.shiftKey) {
                  return;
                }
                if (e.key === "Enter") {
                  e.preventDefault();
                  handleMessageSend();
                }
              }}
            />
            <Button
              className="min-w-16 mr-12"
              onClick={() => {
                handleMessageSend();
              }}
            >
              Send
            </Button>
          </div>
        </div>
      </SheetContent>
    </Sheet>
  );
}
