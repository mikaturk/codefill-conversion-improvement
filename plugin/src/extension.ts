/* --------------------------------------------------------------------------------------------
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See License.txt in the project root for license information.
 * ------------------------------------------------------------------------------------------ */

import * as vscode from 'vscode';
import { workspace, ExtensionContext } from 'vscode';
import * as codeProvider from './codeProvider';

import rand = require('csprng');
import path = require('path');
import fs = require('fs');
import fetch from 'node-fetch';


export function activate(extensionContext: ExtensionContext) {

	if (!extensionContext.globalState.get('codefill-identifier')) {
		extensionContext.globalState.update('codefill-identifier', rand(128, 32));
	}

	const editor = vscode.window.activeTextEditor;

	const codeFillPythonCompletionProvider = new codeProvider.CodeCompletionProvider(extensionContext);
	const selector = 'python';
	// const triggerCharacter = ['.', '('];

	// const completionProvider = vscode.languages.registerCompletionItemProvider(selector, codeFillPythonCompletionProvider, '.')


	extensionContext.subscriptions.push(vscode.languages.registerCompletionItemProvider('python', {
		async provideCompletionItems(document, position, token, context) {
			const val = await callToAPIAndRetrieve(document, extensionContext);
			console.log(val, "123");
			if (!val) return undefined;
			const simpleCompletion = new vscode.CompletionItem(val);
			simpleCompletion.sortText = '0.0.0.0.0';
			return [simpleCompletion];
		}
	}, '.'));
}

function callToAPIAndRetrieve(document: vscode.TextDocument, extensionContext: vscode.ExtensionContext): Promise<string | undefined> {
	return new Promise(resolve => {
		const apiKey = extensionContext!.globalState.get('codefill-identifier');
		var url = "https://codefill-plugin.mikaturk.nl/v1/autocomplete";
		console.log(apiKey);


		fetch(url, {
			method: "POST",
			body: JSON.stringify([document.getText(), ""]),
			headers: {
				'Content-Type': 'application/json',
				'Authorization': 'Basic ' + apiKey
			}
		})
			.then(response => {
				if (!response.ok) {
					console.log("not ok");

					resolve(undefined); return;
				}
				const contentType = response.headers.get('content-type');
				console.log("contentType ", contentType);
				if (!contentType || !contentType.includes('application/json')) { resolve(undefined); return; }
				return response.json();
			})
			.then(json => {
				if (!json.hasOwnProperty('completion')) { resolve(undefined); return; }
				console.log("Resolving ", json.completion);
				resolve(json.completion);
			})
			.catch(() => resolve(undefined));
	});
}

// eslint-disable-next-line @typescript-eslint/no-empty-function
export function deactivate() { }

/**
 * vscode.executeCompletionItemProvider - Execute completion item provider.
	uri - Uri of a text document
	position - A position in a text document
	triggerCharacter - (optional) Trigger completion when the user types the character, like , or (
	itemResolveCount - (optional) Number of completions to resolve (too large numbers slow down completions)
	(returns) - A promise that resolves to a CompletionList-instance.
 */

// const disposable = vscode.commands.registerCommand('codefill-plugin.generateAutoComplete', async () => {

// 	vscode.window.withProgress({
// 		location: vscode.ProgressLocation.Notification,
// 		title: "Generating code completion... Please wait.",
// 		cancellable: true
// 	}, async (progress, token) => {
// 		token.onCancellationRequested(() => {
// 			console.log("Cancelled code completion...");
// 		});

// 		if (editor) {
// 			const filePath = editor.document.uri;
// 			const document = editor.document;
// 			const documentText = document.getText();

// 			const response = await fetch("https://codefill-plugin.mikaturk.nl/v1/autocomplete", {
// 				method: 'POST',
// 				body: JSON.stringify([documentText, ""]),
// 				headers: {
// 					'Content-Type': 'application/json',
// 					'Authorization': 'Basic ' + extensionContext.globalState.get('codefill-identifier')
// 				}
// 			});

// 			const json: any = await response.json();
// 			const jsonText = json.completion;
// 			verifyAccuracy(editor.selection.active.line, jsonText.trim(), extensionContext);

// 			const edit = new vscode.WorkspaceEdit();
// 			edit.insert(filePath, editor.selection.active, jsonText);
// 			workspace.applyEdit(edit);

// 			// editor.edit(editBuilder => {
// 			// 	editBuilder.insert(editor.selection.active, jsonText);
// 			// });
// 		}
// 	});
// });

// extensionContext.subscriptions.push(disposable);

// let isAwaiting = false;
// const provider: vscode.InlineCompletionItemProvider = {
// 	provideInlineCompletionItems: async (document, position, context, token) => {
// 		console.log('provideInlineCompletionItems triggered');

// 		if (isAwaiting) return;

// 		console.log("Sends request");	

// 		if (position.line <= 0) {
// 			return;
// 		}

// 		const documentText = document.getText();

// 		isAwaiting = true;
// 		const response = await fetch("https://codefill-plugin.mikaturk.nl/v1/autocomplete", {
// 			method: 'POST',
// 			body: JSON.stringify([documentText, ""]),
// 			headers: {
// 				'Content-Type': 'application/json',
// 				'Authorization': 'Basic ' + extensionContext.globalState.get('codefill-identifier')
// 			}
// 		});

// 		const json: any = await response.json();
// 		const jsonText = json.completion;
// 		// verifyAccuracy(editor.selection.active.line, jsonText.trim(), extensionContext);

// 		const lineBefore = document.lineAt(position.line - 1).text;
// 		console.log(lineBefore);

// 		const insertText = jsonText;

// 		isAwaiting = false;

// 		return [
// 			{
// 				insertText,
// 				range: new vscode.Range(position.line, 4, position.line, editor!.document.lineAt(position.line).range.end.character),
// 				someTrackingId: 0,
// 			},
// 		] as MyInlineCompletionItem[];

// 	},
// };

// vscode.languages.registerInlineCompletionItemProvider({ pattern: '**' }, provider);

// interface MyInlineCompletionItem extends vscode.InlineCompletionItem {
// 	someTrackingId: number;
// }
// function verifyAccuracy(lineNumber: number, insertedCode: string, context: ExtensionContext) {
// 	const editor = vscode.window.activeTextEditor;

// 	setTimeout(async () => {
// 		const lineText = editor?.document.lineAt(lineNumber).text.trim();
// 		const accepted = (insertedCode?.length != 0) && (lineText === insertedCode);

// 		await fetch("https://codefill-plugin.mikaturk.nl:8443/v1/suggestion_feedback", {
// 			method: 'POST',
// 			body: JSON.stringify(
// 				{
// 					"CodeSuggested": insertedCode,
// 					"CodeAccepted": accepted
// 				}
// 			),
// 			headers: {
// 				'Content-Type': 'application/json',
// 				'Authorization': 'Basic ' + context.globalState.get('codefill-identifier')
// 			}
// 		});
// 	}, 15000);
// }