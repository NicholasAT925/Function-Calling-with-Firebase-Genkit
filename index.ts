import { z, genkit } from 'genkit';
import { vertexAI } from '@genkit-ai/vertexai';
import { gemini20Flash001 } from '@genkit-ai/vertexai';
import { logger } from 'genkit/logging';
import { enableGoogleCloudTelemetry } from '@genkit-ai/google-cloud';
import { startFlowServer } from '@genkit-ai/express';

const ai = genkit({
    plugins: [
        vertexAI({ location: 'us-east1'}),
    ]
});

logger.setLogLevel('debug');
enableGoogleCloudTelemetry();


export const MenuItemSchema = z.object({
    title: z.string()
        .describe('The name of the menu item'),
    description: z.string()
        .describe('Details, including ingredients and preparation'),
    price: z.number()
        .describe('Price, in dollars'),
});

export type MenuItem = z.infer<typeof MenuItemSchema>;

// Input schema for a question about the menu
export const MenuQuestionInputSchema = z.object({
    question: z.string(),
});

// Output schema containing an answer to a question
export const AnswerOutputSchema = z.object({
    answer: z.string(),
});


const menuData: Array<MenuItem> = require('../data/menu.json')

export const menuTool = ai.defineTool(
{
    name: 'todaysMenu',
    description: "Use this tool to retrieve all the items on today's menu",
    inputSchema: z.object({}),
    outputSchema: z.object({
        menuData: z.array(MenuItemSchema)
            .describe('A list of all the items on the menu'),
    }),
},
async () => Promise.resolve({ menuData: menuData })
);


export const dataMenuPrompt = ai.definePrompt(
    {
        name: 'dataMenu',
        model: gemini20Flash001,
        input: { schema: MenuQuestionInputSchema },
        output: { format: 'text' },
        tools: [menuTool],
    },
    `
    You are acting as a helpful AI assistant named Walt that can answer
    questions about the food available on the menu at Walt's Burgers.
    
    Answer this customer's question, in a concise and helpful manner,
    as long as it is about food on the menu or something harmless like sports.
    Use the tools available to answer food and menu questions.
    DO NOT INVENT ITEMS NOT ON THE MENU.
    
    Question:
    {{question}} ?
    `
    );

    
export const menuQuestionFlow = ai.defineFlow(
    {
        name: 'menuQuestion',
        inputSchema: MenuQuestionInputSchema,
        outputSchema: AnswerOutputSchema,
    },
    async (input) => {
        const response = await dataMenuPrompt({
            question: input.question,
        });
        return { answer: response.text };
    }
);


startFlowServer({
    flows: [menuQuestionFlow],
    port: 8080,
    cors: {
        origin: '*',
    },
});