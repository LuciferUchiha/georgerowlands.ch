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

export default function ChatBot() {
  const [apiKey, setApiKey] = useState("");
  const [model, setModel] = useState("gpt-4o");
  const [inputValue, setInputValue] = useState("");

  useEffect(() => {
    const value = localStorage.getItem("apiKey") || ""
    setApiKey(value)
  }, [])

  const saveKeyToLocalStorage = (key: string) => {
    localStorage.setItem("apiKey", key)
  }

  const embed_text = (text: string) => {
    axios
      .post(
        "https://api.openai.com/v1/embeddings",
        {
          input: text,
          model: "text-embedding-3-small",
        },
        {
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${apiKey}`,
          },
        }
      )
      .then((response) => {
        console.log(response.data);
      })
      .catch((error) => {
        console.error(error);
      });
  };

  const send_message = (text: string) => {
    axios
      .post(
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
              content: text,
            },
          ]
        },
        {
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${apiKey}`,
          },
        }
      )
      .then((response) => {
        console.log(response.data.choices[0].text);
      })
      .catch((error) => {
        console.error(error);
      });
  }

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
                embed_text("What is the capital of France?");
                send_message(inputValue);
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
