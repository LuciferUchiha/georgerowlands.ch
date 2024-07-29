import { LuBrainCircuit, LuInfo } from "react-icons/lu";
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
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "~/components/ui/tooltip";
import { Input } from "~/components/ui/input";
import { Label } from "~/components/ui/label";
import { Textarea } from "~components/ui/textarea";
import { Button } from "~components/ui/button";
import axios from "axios";
import { useState, useEffect } from "react";

type Role = "system" | "user" | "assistant";

type Message = {
  role: Role;
  content: string;
};

type Context = {
  id : string;
  page_content : string;
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
        className={`p-4 max-w-[80%] rounded-lg ${
          role === "assistant"
            ? "bg-zinc-800 text-zinc-50"
            : "bg-zinc-50 text-zinc-800"
        }`}
      >
        {content}
      </div>
    </div>
  );
}

const MessageList = ({ messages }: { messages: Message[] }) => {
  // scrollable div
  return (
    <div className="flex flex-col overflow-y-auto h-[calc(100vh-300px)]">
      {messages.map((message, index) => (
        <MessageCard key={index} role={message.role} content={message.content} />
      ))}
    </div>
  );
}

const getPrompt = (question: string, contexts: Context[]) => {
  const parsedContexts = contexts.map((context) => {
    return `[${context.title}](${context.path}):\n${context.page_content}\n\n`;
  });
  const context = parsedContexts.join("");

  return `Please answer the following question, using the following documents. If
  any of the documents where helpful, please include their links in the references.
  If the question is just general conversation or chit-chat do not use or include any 
  references.

  Documents:
  ${context}

  Question:
  ${question}

  Answer:`;
}

export default function ChatBot() {
  const [threshold, setThreshold] = useState<number>(0.375);
  const [apiKey, setApiKey] = useState<string>("");
  const [model, setModel] = useState<string>("gpt-4o");
  const [inputValue, setInputValue] = useState<string>("");
  const [messages, setMessages] = useState<Message[]>([]);

  useEffect(() => {
    const value = localStorage.getItem("apiKey") || ""
    setApiKey(value)
  }, [])

  const saveKeyToLocalStorage = (key: string) => {
    localStorage.setItem("apiKey", key)
  }

  const addMessage = (role: Role, content: string) => {
    setMessages((messages) => [...messages, { role, content }]);
  }

  const updateLastMessage = (content: string) => {
    setMessages((messages) => {
      const newMessages = [...messages];
      newMessages[messages.length - 1].content += content;
      return newMessages;
    });
  }

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
          topK: 5,
          includeMetadata: true,
        },
        {
          headers: {
            "Api-Key": process.env.NEXT_PUBLIC_PINECONE_KEY || "",
          },
        }
      );
      const matches = contextResponse.data.matches;
      const contexts = matches.map((match: { id: string; score:number, metadata: any }) => {
        const title = match.metadata["Header 1"];
        const cleanPath = match.metadata.path.replaceAll("../pages", "").replaceAll(".mdx", "");
        return {
          id: match.id,
          page_content: match.metadata.page_content,
          path: cleanPath,
          title: title,
          score: match.score,
        };
      });
      console.log("contexts", contexts);
      const filteredContexts = contexts.filter((context) => context.score > threshold);
      console.log("filteredContexts", filteredContexts);

      // all messages minus the last one
      const prevMessages = messages.slice(0, messages.length);
      // Step 3: Send the message along with the context
      const messageResponse = await fetch('https://api.openai.com/v1/chat/completions', {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${apiKey}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model: model,
          messages: [
            {
              role: "system",
              content: "You are a helpful assistant.",
            },
            ...prevMessages,
            {
              role: "user",
              content: getPrompt(question, filteredContexts),
            },
          ],
          stream: true,
        }),
      });
      const reader = messageResponse.body?.pipeThrough(new TextDecoderStream()).getReader();
      if (!reader) throw new Error("Failed to create a reader");
      addMessage("assistant", "");
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        let dataDone = false;
        const arr = value.split('\n');
        arr.forEach((data) => {
          if (data.length === 0) return; 
          if (data.startsWith(':')) return; // ignore sse comment message
          if (data === 'data: [DONE]') {
            dataDone = true;
            return;
          }
          const json = JSON.parse(data.substring(6));
          console.log(json);
          const delta = json.choices[0].delta.content;
          if (delta)
            updateLastMessage(delta);
        });
        if (dataDone) break;
      }
    } catch (error) {
      console.error(error);
      addMessage("assistant", "Sorry, there was an error processing your request.");
    }
    console.log(messages);
  };

  return (
    <Sheet>
      <SheetTrigger>
        <LuBrainCircuit className="w-[24px] h-[24px]" />
      </SheetTrigger>
      <SheetContent side="right" className="min-w-[800px] dark:border-zinc-800">
        <SheetHeader>
          <SheetTitle>Talk to the Garden</SheetTitle>
          <SheetDescription>
            Using Retrieval Augmented Generation (RAG) you can talk to the
            garden and ask questions.
          </SheetDescription>
        </SheetHeader>
        <div className="mt-4">
          <div className="flex flex-row justify-between">
            <div className="flex flex-col w-96">
              <div className="flex flex-row">
                <Label htmlFor="apiKey" className="mb-2 mr-2">
                  OpenAI API Key
                </Label>
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger className="flex flex-col">
                      <LuInfo />
                    </TooltipTrigger>
                    <TooltipContent>
                      <p>
                        Your OpenAI API key is required to use the chatbot. You
                        can create an API key here. <br /> The key you provide
                        will not be sent or stored on any servers, only in your
                        browser.
                      </p>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
              </div>
              <Input
                id="apiKey"
                placeholder="sk-********"
                value={apiKey}
                onChange={(e) => {
                  setApiKey(e.target.value)
                  saveKeyToLocalStorage(e.target.value)
                }}
              />
            </div>
            <div className="flex flex-col w-32">
              <Label className="mb-2">Threshold</Label>
              <Input
                type="number"
                defaultValue={threshold}
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
                  <SelectItem value="gpt-4o">gpt-4o</SelectItem>
                  <SelectItem value="gpt-4o-mini">gpt-4o-mini</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>
          <div className="mt-4">
            <MessageList messages={messages} />
          </div>
          <div className="fixed bottom-0 my-6 gap-4 w-full flex flex-row">
            <Textarea
              className="resize-none text-base w-[640px]"
              placeholder="Send a message..."
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  addMessage("user", inputValue);
                  rag(inputValue);
                  setInputValue("");
                }
              }}
            />
            <Button
              className="w-24"
              onClick={() => {
                addMessage("user", inputValue);
                rag(inputValue);
                setInputValue("");
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