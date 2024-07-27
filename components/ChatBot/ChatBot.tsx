import { LuBrainCircuit } from "react-icons/lu";
import { LuInfo } from "react-icons/lu";
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

export default function ChatBot() {
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
            <div className="flex flex-col w-48 mr-4">
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
                      Your OpenAI API key is required to use the chatbot.
                      You can create an API key here. <br /> The key you provide will not 
                      be sent or stored on any servers, only in your browser.
                      </p>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
              </div>
              <Input id="apiKey" placeholder="sk-********" />
            </div>
            <div className="flex flex-col w-48">
              <Label className="mb-2">Model</Label>
              <Select defaultValue="gpt-4o">
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
        </div>
      </SheetContent>
    </Sheet>
  );
}
