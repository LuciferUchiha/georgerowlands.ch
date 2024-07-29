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
import { get } from "http";

type Role = "system" | "user" | "assistant";

type Message = {
  role: Role;
  content: string;
};

type Embedding = number[];

type Context = {
  id : string;
  page_content : string;
};

export default function ChatBot() {
  const [apiKey, setApiKey] = useState<string>("");
  const [model, setModel] = useState<string>("gpt-4o");
  const [inputValue, setInputValue] = useState<string>("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [queryEmbedding, setQueryEmbedding] = useState<Embedding>();
  const [context, setContext] = useState<Context[]>([]);

  useEffect(() => {
    const value = localStorage.getItem("apiKey") || ""
    setApiKey(value)
  }, [])

  const saveKeyToLocalStorage = (key: string) => {
    localStorage.setItem("apiKey", key)
  }

  const add_message = (role: Role, content: string) => {
    setMessages([...messages, { role: role, content: content}]);
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
          top_k: 5,
        },
        {
          headers: {
            "Content-Type": "application/json",
            "Api-Key": process.env.NEXT_PUBLIC_PINECONE_KEY,
            "X-Pinecone-API-Version": "2024-07",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Credentials": "true",
            "Access-Control-Allow-Methods": "GET,HEAD,OPTIONS,POST,PUT",
            "Access-Control-Allow-Headers": "Access-Control-Allow-Headers, Origin,Accept, X-Requested-With, Content-Type, Access-Control-Request-Method, Access-Control-Request-Headers, Api-Key, X-Pinecone-API-Version",
          },
        }
      );
      const matches = contextResponse.data.matches;
      const contexts = matches.map((match: { id: string; page_content: string }) => {
        return {
          id: match.id,
          page_content: match.page_content,
        };
      });
      console.log("contexts", contexts);

      // Step 3: Send the message along with the context
      const messageResponse = await axios.post(
        "https://api.openai.com/v1/chat/completions",
        {
          model: model,
          messages: [
            {
              role: "system",
              content: "You are a helpful assistant.",
            },
            {
              role: "user",
              content: question,
            },
          ],
        },
        {
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${apiKey}`,
          },
        }
      );
      const answer = messageResponse.data.choices[0].text;
      console.log("answer", answer);
      add_message("assistant", answer);
    } catch (error) {
      console.error(error);
      add_message("assistant", "Sorry, there was an error processing your request.");
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
            <div className="flex flex-col w-96 mr-4">
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
          <div className="fixed bottom-0 my-6 gap-4 w-full flex flex-row">
            <Textarea
              className="resize-none text-base w-[640px]"
              placeholder="Send a message..."
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
            />
            <Button
              className="w-24"
              onClick={() => {
                add_message("user", inputValue);
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
