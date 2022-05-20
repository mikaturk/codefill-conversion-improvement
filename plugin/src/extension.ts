/* --------------------------------------------------------------------------------------------
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See License.txt in the project root for license information.
 * ------------------------------------------------------------------------------------------ */

import * as vscode from 'vscode';
import { ExtensionContext } from 'vscode';

import rand = require('csprng');
import path = require('path');
import fs = require('fs');
import fetch from 'node-fetch';

export function activate(extensionContext: ExtensionContext) {

	if (!extensionContext.globalState.get('codefill-identifier')) {
		extensionContext.globalState.update('codefill-identifier', rand(128, 32));
	}

	extensionContext.subscriptions.push(vscode.languages.registerCompletionItemProvider('python', {
		async provideCompletionItems(document, position, token, context): Promise<vscode.CompletionItem[]> {
			const val = await callToAPIAndRetrieve(document, extensionContext);
			if (!val) return [];
			const simpleCompletion = new vscode.CompletionItem(val);
			simpleCompletion.sortText = '0.0.0.0.0';
			return [simpleCompletion];
		}
	}, '.', ' ', ':', ',', '[', '(', '{', '~', '+', '/', '*', '-', '!', '&', '&&', '|', '||', '^', '**'))
}

async function callToAPIAndRetrieve(document: vscode.TextDocument, extensionContext: vscode.ExtensionContext): Promise<string | undefined> {
	const editor = vscode.window.activeTextEditor;

	if (!editor) return undefined;

	const currPos = editor.selection.active;
	const startPos = new vscode.Position(currPos.line, currPos.character - 1);
	const endPos = new vscode.Position(currPos.line, currPos.character);
	const range = new vscode.Range(startPos, endPos);
	const character = editor.document.getText(range);
	console.log("Char = ", character);

	const line = document.lineAt(currPos.line);
	const lineSplit = line.text.match(/[\w]+/g);
	const lastWord = lineSplit?.pop();
	console.log("lastWord = ", lastWord);

	// DOT, AWAIT, ASSERT, RAISE, DEL, LAMBDA, YIELD, RETURN,
	// EXCEPT, WHILE, FOR, IF, ELIF, ELSE, GLOBAL, IN, AND, NOT,
	// OR, IS, BINOP, WITH

	const allowedCharacters = ['.', ' ', ':', ',', '[', '(', '{', '~', '+', '/', '*', '-', '!', '&', '&&', '|', '||', '^', '**'];

	if (!allowedCharacters.includes(character)) return undefined;

	try {
		const apiKey = extensionContext!.globalState.get('codefill-identifier');
		var url = "https://codefill-plugin.mikaturk.nl/v1/autocomplete";

		const response = await fetch(url, {
			method: "POST",
			body: JSON.stringify([document.getText(), ""]),
			headers: {
				'Content-Type': 'application/json',
				'Authorization': 'Basic ' + apiKey
			}
		})

		if (!response.ok) {
			return undefined;
		}

		const contentType = response.headers.get('content-type');
		if (!contentType || !contentType.includes('application/json')) {
			return undefined;
		}

		const json = await response.json();
		if (!json.hasOwnProperty('completion')) { 
			return undefined;
		}
		console.log("Resolving ", json.completion);
		return json.completion;
	} catch (e) {
		return undefined;
	}
}

// eslint-disable-next-line @typescript-eslint/no-empty-function
export function deactivate() { }

function verifyAccuracy(lineNumber: number, insertedCode: string, context: ExtensionContext) {
	const editor = vscode.window.activeTextEditor;

	setTimeout(async () => {
		const lineText = editor?.document.lineAt(lineNumber).text.trim();
		const accepted = (insertedCode?.length != 0) && (lineText === insertedCode);

		await fetch("https://codefill-plugin.mikaturk.nl:8443/v1/suggestion_feedback", {
			method: 'POST',
			body: JSON.stringify(
				{
					"CodeSuggested": insertedCode,
					"CodeAccepted": accepted
				}
			),
			headers: {
				'Content-Type': 'application/json',
				'Authorization': 'Basic ' + context.globalState.get('codefill-identifier')
			}
		});
	}, 15000);
}