/* --------------------------------------------------------------------------------------------
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See License.txt in the project root for license information.
 * ------------------------------------------------------------------------------------------ */

import * as vscode from 'vscode';
import { ExtensionContext } from 'vscode';

import rand from 'csprng';
import fetch from 'node-fetch';
import { v4 as uuidv4 } from 'uuid';

export function activate(extensionContext: ExtensionContext) {

	console.log("RUNNING");

	if (!extensionContext.globalState.get('codefill-uuid')) {
		extensionContext.globalState.update('codefill-uuid', rand(128, 16));
	}

	extensionContext.subscriptions.push(vscode.commands.registerCommand('verifyInsertion', verifyInsertion));

	const codeForMeCompletionProvider = extensionContext.subscriptions.push(vscode.languages.registerCompletionItemProvider('python', {
		async provideCompletionItems(document, position, token, context) {
			const jsonResponse = await callToAPIAndRetrieve(document, position, extensionContext);
			if (!jsonResponse) return undefined;
			const completion = jsonResponse.completion;
			if (completion == "") {
				console.log("empty string");

				return undefined;
			}
			console.log("Completion ", completion);

			const completionToken = jsonResponse.completionToken;
			console.log("CompletionToken", completionToken);

			const apiKey = extensionContext.globalState.get('codefill-uuid');
			const completionItem = new vscode.CompletionItem('\u276E\uff0f\u276f: ' + completion);
			completionItem.sortText = '0.0000';
			completionItem.insertText = completion;
			completionItem.command = {
				command: 'verifyInsertion',
				title: 'Verify Insertion',
				arguments: [position, completion, context, completionToken, apiKey]
			};
			return [completionItem];
		}
	}, '.', ' ', ',', '[', '(', '{', '~', '+', '/', '*', '-', '!', '&', '&&', '|', '||', '^', '**'));
}

async function callToAPIAndRetrieve(document: vscode.TextDocument, position: vscode.Position, extensionContext: vscode.ExtensionContext): Promise<any | undefined> {
	const editor = vscode.window.activeTextEditor;
	if (!editor) return undefined;

	const startPos = new vscode.Position(position.line, position.character - 1);
	const endPos = new vscode.Position(position.line, position.character);
	const range = new vscode.Range(startPos, endPos);
	const character = document.getText(range);
	const line = document.lineAt(position.line).text;
	
	if (position.character !== line.length) return undefined;
	
	console.log("Char = ", character);

	const startPosLine = new vscode.Position(position.line, 0);
	const endPosLine = new vscode.Position(position.line, position.character);
	const rangeLine = new vscode.Range(startPosLine, endPosLine);

	const lineSplit = document.getText(rangeLine).match(/[\w]+/g);
	const lastWord = lineSplit?.pop();
	console.log("lastWord = ", lastWord);

	// DOT, AWAIT, ASSERT, RAISE, DEL, LAMBDA, YIELD, RETURN,
	// EXCEPT, WHILE, FOR, IF, ELIF, ELSE, GLOBAL, IN, AND, NOT,
	// OR, IS, BINOP, WITH

	const allowedCharacters = ['.', ' ', ':', ',', '[', '(', '{', '~', '+', '/', '*', '-', '!', '&', '&&', '|', '||', '^', '**'];

	if (character !== ' ' && !allowedCharacters.includes(character.trim())) return undefined;

	const documentLineCount = document.lineCount - 1;
	const lastLine = document.lineAt(documentLineCount);
	const beginDocumentPosition = new vscode.Position(0, 0);
	const firstHalfRange = new vscode.Range(beginDocumentPosition, position);

	const lastLineCharacterOffset = lastLine.range.end.character;
	const lastLineLineOffset = lastLine.lineNumber;
	const endDocumentPosition = new vscode.Position(lastLineLineOffset, lastLineCharacterOffset);
	const secondHalfRange = new vscode.Range(position, endDocumentPosition);

	const firstHalf = editor.document.getText(firstHalfRange);
	const secondHalf = editor.document.getText(secondHalfRange);

	const triggerPoint = getTriggerPoint(lastWord, character);
	console.log("tp: ", triggerPoint);

	try {
		const apiKey = extensionContext!.globalState.get('codefill-uuid');
		const url = "https://code4me.me/api/v1/autocomplete";

		const response = await fetch(url, {
			method: "POST",
			body: JSON.stringify(
				{
					"parts": [firstHalf, secondHalf],
					"triggerPoint": null,
					"language": "python"
				}
			),
			headers: {
				'Content-Type': 'application/json',
				'Authorization': 'Bearer ' + apiKey
			}
		});

		if (!response.ok) {
			console.error("Response status not OK! Status: ", response.status);
			return undefined;
		}

		const contentType = response.headers.get('content-type');
		if (!contentType || !contentType.includes('application/json')) {
			console.error("Wrong content type!");
			return undefined;
		}

		const json = await response.json();
		if (!Object.prototype.hasOwnProperty.call(json, 'completion')) {
			console.error("Completion field not found in response!");
			return undefined;
		}
		return json;
	} catch (e) {
		console.error("Unexpected error: ", e);
		return undefined;
	}
}

// eslint-disable-next-line @typescript-eslint/no-empty-function
export function deactivate() { }

function verifyInsertion(position: vscode.Position, completion: string, extensionContext: ExtensionContext, completionToken: string, apiKey: string) {
	const editor = vscode.window.activeTextEditor;
	const document = editor!.document;
	let lineNumber = position.line;
	let characterOffset = position.character;

	const listener = vscode.workspace.onDidChangeTextDocument(event => {
		for (const changes of event.contentChanges) {
			console.log(changes);

			const text = changes.text;
			const changedLineNumber = changes.range.start.line;
			if (changedLineNumber == lineNumber) {
				if (changes.range.end.character < characterOffset + 1) {
					if (changes.text === '') {
						characterOffset -= changes.rangeLength;
					} if (changes.text.includes('\n')) {
						characterOffset = 0;
						lineNumber += (text.match(/\n/g) || []).length;
					} else {
						characterOffset += changes.text.length;
					}
				}
			} else if (lineNumber == 0 || changedLineNumber < lineNumber) {
				lineNumber += (text.match(/\n/g) || []).length;
			}

			if (changes.range.end.line < lineNumber) {
				if (changes.text === '') {
					lineNumber -= changes.range.end.line - changedLineNumber;
				}
			}
		}

		const editor = vscode.window.activeTextEditor;

		if (!editor) return undefined;

		console.log("line = ", editor.document.lineAt(lineNumber).text);
		console.log("Line loc = ", lineNumber);
		const startPos = new vscode.Position(lineNumber, characterOffset);
		const endPos = new vscode.Position(lineNumber, characterOffset + 1);
		const range = new vscode.Range(startPos, endPos);
		const character = document.getText(range);
		console.log("Char = ", character);
		console.log("Char loc = ", characterOffset);
	});



	setTimeout(async () => {
		listener.dispose();
		console.log("send");
		const lineText = editor?.document.lineAt(lineNumber).text;
		console.log(lineText?.substring(characterOffset));
		
		fetch("https://code4me.me/api/v1/completion", {
			method: 'POST',
			body: JSON.stringify(
				{
					"completionToken": completionToken,
					"completion": completion,
					"line": lineText?.substring(characterOffset).trim
				}
			),
			headers: {
				'Content-Type': 'application/json',
				'Authorization': 'Bearer ' + apiKey
			}
		});
	}, 15000);
}

function getTriggerPoint(lastWord: string | undefined, character: string) {
	if (lastWord == undefined) return undefined;
	if (character == " ") return lastWord;
	return character;
}
