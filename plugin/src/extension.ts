/* --------------------------------------------------------------------------------------------
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See License.txt in the project root for license information.
 * ------------------------------------------------------------------------------------------ */

import * as vscode from 'vscode';
import { ExtensionContext } from 'vscode';
import rand from 'csprng';
import fetch from 'node-fetch';

export function activate(extensionContext: ExtensionContext) {
	if (!extensionContext.globalState.get('codefill-uuid')) {
		extensionContext.globalState.update('codefill-uuid', rand(128, 16));
	}

	const codeFillUuid: string = extensionContext.globalState.get('codefill-uuid')!;

	extensionContext.subscriptions.push(vscode.commands.registerCommand('verifyInsertion', verifyInsertion));

	extensionContext.subscriptions.push(vscode.languages.registerCompletionItemProvider('python', {
		async provideCompletionItems(document, position, token, context) {
			const jsonResponse = await callToAPIAndRetrieve(document, position, codeFillUuid);
			if (!jsonResponse) return undefined;

			const listPredictionItems = jsonResponse.predictions;
			const completionToken = jsonResponse.verifyToken;
			if (listPredictionItems.length == 0 || !completionToken) return undefined;
			
			return listPredictionItems.map((prediction: string) => {
				const completionItem = new vscode.CompletionItem('\u276E\uff0f\u276f: ' + prediction);
				completionItem.sortText = '0.0000';
				completionItem.insertText = prediction;
				completionItem.command = {
					command: 'verifyInsertion',
					title: 'Verify Insertion',
					arguments: [position, prediction, completionToken, codeFillUuid]
				};
				return completionItem;
			});
		}
	}, ' ', '.', '+', '-', '*', '/', '%', '*', '<', '>', '&', '|', '^', '=', '!', ';', ',', '[', '(', '{', '~'));
}

async function callToAPIAndRetrieve(document: vscode.TextDocument, position: vscode.Position, codeFillUuid: string): Promise<any | undefined> {
	const editor = vscode.window.activeTextEditor;
	if (!editor) return undefined;

	const startDoubleCharacterPos = new vscode.Position(position.line, position.character - 2);
	const endPos = new vscode.Position(position.line, position.character);
	const rangeDoubleCharacter = new vscode.Range(startDoubleCharacterPos, endPos);

	const startSingleCharacterPos = new vscode.Position(position.line, position.character - 1);
	const rangeSingleCharacter = new vscode.Range(startSingleCharacterPos, endPos);

	const singleCharacter = document.getText(rangeSingleCharacter);
	const doubleCharacter = document.getText(rangeDoubleCharacter).trim();

	const line = document.lineAt(position.line).text;

	if (position.character !== line.length) return undefined;

	const startPosLine = new vscode.Position(position.line, 0);
	const endPosLine = new vscode.Position(position.line, position.character);
	const rangeLine = new vscode.Range(startPosLine, endPosLine);

	const lineSplit = document.getText(rangeLine).match(/[\w]+/g);
	const lastWord = lineSplit?.pop();

	// DOT, AWAIT, ASSERT, RAISE, DEL, LAMBDA, YIELD, RETURN,
	// EXCEPT, WHILE, FOR, IF, ELIF, ELSE, GLOBAL, IN, AND, NOT,
	// OR, IS, BINOP, WITH

	const allowedCharacters = ['.', '+', '-', '*', '/', '%', '**', '<<', '>>', '&', '|', '^', '+=', '-=', '==', '!=', ';', ',', '[', '(', '{', '~', '='];

	let triggerCharacter = doubleCharacter;
	if (!allowedCharacters.includes(doubleCharacter)) {
		triggerCharacter = singleCharacter;
		if (!allowedCharacters.includes(singleCharacter)) return undefined;
	}

	const documentLineCount = document.lineCount - 1;
	const lastLine = document.lineAt(documentLineCount);
	const beginDocumentPosition = new vscode.Position(0, 0);
	const leftRange = new vscode.Range(beginDocumentPosition, position);

	const lastLineCharacterOffset = lastLine.range.end.character;
	const lastLineLineOffset = lastLine.lineNumber;
	const endDocumentPosition = new vscode.Position(lastLineLineOffset, lastLineCharacterOffset);
	const rightRange = new vscode.Range(position, endDocumentPosition);

	const leftText = editor.document.getText(leftRange);
	const rightText = editor.document.getText(rightRange);

	const triggerPoint = getTriggerPoint(lastWord, triggerCharacter);

	try {
		const url = "https://code4me.me/api/v1/prediction/autocomplete";

		const response = await fetch(url, {
			method: "POST",
			body: JSON.stringify(
				{
					"leftContext": leftText.substring(-1024),
					"rightContext": rightText.substring(0, 1024),
					"triggerPoint": triggerPoint,
					"language": "python",
					"ide": "vsc"
				}
			),
			headers: {
				'Content-Type': 'application/json',
				'Authorization': 'Bearer ' + codeFillUuid
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
		
		if (!Object.prototype.hasOwnProperty.call(json, 'predictions')) {
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

function verifyInsertion(position: vscode.Position, completion: string, completionToken: string, apiKey: string) {
	const editor = vscode.window.activeTextEditor;
	const document = editor!.document;
	const documentName = document.fileName;
	let lineNumber = position.line;
	const originalOffset = position.character;
	let characterOffset = originalOffset;
	const listener = vscode.workspace.onDidChangeTextDocument(event => {
		if (vscode.window.activeTextEditor == undefined) return;
		if (vscode.window.activeTextEditor.document.fileName !== documentName) return;
		for (const changes of event.contentChanges) {
			const text = changes.text;
			const startChangedLineNumber = changes.range.start.line;
			const endChangedLineNumber = changes.range.end.line;

			if (startChangedLineNumber == lineNumber - 1 && endChangedLineNumber == lineNumber && changes.text == '') {
				lineNumber--;
				const startLine = document.lineAt(startChangedLineNumber);
				if (startLine.isEmptyOrWhitespace) characterOffset++;
				else characterOffset += changes.range.start.character + 1;
			}

			if (startChangedLineNumber == lineNumber) {
				if (changes.range.end.character < characterOffset + 1) {
					if (changes.text === '') {
						characterOffset -= changes.rangeLength;
					} if (changes.text.includes('\n')) {
						characterOffset = originalOffset;
						lineNumber += (text.match(/\n/g) ?? []).length;
					} else {
						characterOffset += changes.text.length;
					}
				}
			} else if (lineNumber == 0 || startChangedLineNumber < lineNumber) {
				lineNumber += (text.match(/\n/g) ?? []).length;
			}

			if (changes.range.end.line <= lineNumber) {
				if (changes.text === '') {
					lineNumber -= changes.range.end.line - startChangedLineNumber;
				}
			}
		}
	});

	setTimeout(async () => {
		listener.dispose();
		const lineText = editor?.document.lineAt(lineNumber).text;

		fetch("https://code4me.me/api/v1/prediction/verify", {
			method: 'POST',
			body: JSON.stringify(
				{
					"verifyToken": completionToken,
					"chosenPrediction": completion,
					"line": lineText?.substring(characterOffset).trim
				}
			),
			headers: {
				'Content-Type': 'application/json',
				'Authorization': 'Bearer ' + apiKey
			}
		});
	}, 30000);
}

function getTriggerPoint(lastWord: string | undefined, character: string) {
	if (lastWord == undefined) return null;
	if (character == " ") return lastWord;
	return character.trim();
}
