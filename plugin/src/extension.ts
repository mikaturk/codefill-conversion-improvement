/* --------------------------------------------------------------------------------------------
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See License.txt in the project root for license information.
 * ------------------------------------------------------------------------------------------ */

import * as vscode from 'vscode';
import { ExtensionContext } from 'vscode';

import rand from 'csprng';
import fetch from 'node-fetch';
import {v4 as uuidv4} from 'uuid';

export function activate(extensionContext: ExtensionContext) {

	if (!extensionContext.globalState.get('codefill-uuid')) {
		extensionContext.globalState.update('codefill-uuid', uuidv4());
	}

	extensionContext.subscriptions.push(vscode.commands.registerCommand('verifyInsertion', verifyInsertion));

	const codeForMeCompletionProvider = extensionContext.subscriptions.push(vscode.languages.registerCompletionItemProvider('python', {
		async provideCompletionItems(document, position, token, context) {
			const jsonResponse = await callToAPIAndRetrieve(document, extensionContext);
			if (!jsonResponse) return undefined;
			const completion = jsonResponse.completion;
			const completionToken = jsonResponse.completionToken; 
			const completionItem = new vscode.CompletionItem('\u276E\uff0f\u276f: ' + completion);
			completionItem.sortText = '0.0000';
			completionItem.insertText = completion;
			completionItem.command = {
				command: 'verifyInsertion',
				title: 'Verify Insertion',
				arguments: [position.line, completion, context, completionToken]
			};
			return [completion];
		}
	}, '.', ' ', ',', '[', '(', '{', '~', '+', '/', '*', '-', '!', '&', '&&', '|', '||', '^', '**'));
}

async function callToAPIAndRetrieve(document: vscode.TextDocument, extensionContext: vscode.ExtensionContext): Promise<any | undefined> {
	const editor = vscode.window.activeTextEditor;

	if (!editor) return undefined;

	const currPos = editor.selection.active;
	const startPos = new vscode.Position(currPos.line, currPos.character - 1);
	const endPos = new vscode.Position(currPos.line, currPos.character);
	const range = new vscode.Range(startPos, endPos);
	const character = document.getText(range);
	console.log("Char = ", character);

	const line = document.lineAt(currPos.line);
	const lineSplit = line.text.match(/[\w]+/g);
	const lastWord = lineSplit?.pop();
	console.log("lastWord = ", lastWord);

	// DOT, AWAIT, ASSERT, RAISE, DEL, LAMBDA, YIELD, RETURN,
	// EXCEPT, WHILE, FOR, IF, ELIF, ELSE, GLOBAL, IN, AND, NOT,
	// OR, IS, BINOP, WITH

	const allowedCharacters = ['.', ' ', ':', ',', '[', '(', '{', '~', '+', '/', '*', '-', '!', '&', '&&', '|', '||', '^', '**'];

	if (!allowedCharacters.includes(character)) {
		console.log("rejected");
		
		return undefined;
	}

	try {
		const apiKey = extensionContext!.globalState.get('codefill-uuid');
		const url = "https://code4me.me/api/v1/autocomplete";

		const response = await fetch(url, {
			method: "POST",
			// body: JSON.stringify([document.getText(), ""]),
			body: JSON.stringify(
				{
					"parts": [document.getText(), ""],
					"triggerPoint": null,
					"language": 'python'
				}
			),
			headers: {
				'Content-Type': 'application/json',
				'Authorization': 'Bearer ' + apiKey
			}
		});

		if (!response.ok) {
			return undefined;
		}

		const contentType = response.headers.get('content-type');
		if (!contentType || !contentType.includes('application/json')) {
			return undefined;
		}

		const json = await response.json();
		if (!Object.prototype.hasOwnProperty.call(json, 'completion')) {
			return undefined;
		}
		return json;
	} catch (e) {
		return undefined;
	}
}

// eslint-disable-next-line @typescript-eslint/no-empty-function
export function deactivate() { }

function verifyInsertion(lineNumber: number, completion: string, context: ExtensionContext, completionToken: string) {
	const editor = vscode.window.activeTextEditor;

	const listener = vscode.workspace.onDidChangeTextDocument(event => {
		for (const changes of event.contentChanges) {
			console.log(changes);

			const text = changes.text;
			if (changes.range.start.line < lineNumber) {
				lineNumber += (text.match(/\n/g) || []).length;
			}

			if (changes.range.end.line < lineNumber) {
				if (changes.text === '') {
					lineNumber -= changes.range.end.line - changes.range.start.line;
				}
			}
		}

		const editor = vscode.window.activeTextEditor;

		if (!editor) return undefined;

		console.log("line = ", editor.document.lineAt(lineNumber).text);
		console.log("Line loc = ", lineNumber);
	});


	setTimeout(async () => {
		listener.dispose();
		const lineText = editor?.document.lineAt(lineNumber).text.trim();
		await fetch("https://code4me.me/api/v1/completion", {
			method: 'POST',
			body: JSON.stringify(
				{
					"completionToken": completionToken,
					"completion": completion,
					"line": lineText
				}
			),
			headers: {
				'Content-Type': 'application/json',
				'Authorization': 'Bearer ' + context.globalState.get('codefill-uuid')
			}
		});
	}, 15000);
}