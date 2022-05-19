/* --------------------------------------------------------------------------------------------
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See License.txt in the project root for license information.
 * ------------------------------------------------------------------------------------------ */

import * as vscode from 'vscode';
import { workspace, ExtensionContext } from 'vscode';
import fetch from 'node-fetch';

import rand = require('csprng');
import path = require('path');
import fs = require('fs');

export function activate(extensionContext: ExtensionContext) {

	const disposable2 = vscode.commands.registerCommand(
		'extension.inline-completion-settings',
		() => {
			vscode.window.showInformationMessage('Show settings');
		}
	);

	extensionContext.subscriptions.push(disposable2);


	if (!extensionContext.globalState.get('codefill-identifier')) {
		extensionContext.globalState.update('codefill-identifier', rand(128, 32));
	}

	const editor = vscode.window.activeTextEditor;

	const disposable = vscode.commands.registerCommand('codefill-plugin.generateAutoComplete', async () => {

		vscode.window.withProgress({
			location: vscode.ProgressLocation.Notification,
			title: "Generating code completion... Please wait.",
			cancellable: true
		}, async (progress, token) => {
			token.onCancellationRequested(() => {
				console.log("Cancelled code completion...");
			});

			if (editor) {
				const filePath = editor.document.uri;
				const document = editor.document;
				const documentText = document.getText();

				const response = await fetch("https://codefill-plugin.mikaturk.nl:8443/v1/autocomplete", {
					method: 'POST',
					body: JSON.stringify([documentText, ""]),
					headers: {
						'Content-Type': 'application/json',
						'Authorization': 'Basic ' + extensionContext.globalState.get('codefill-identifier')
					}
				});

				const json: any = await response.json();
				const jsonText = json.completion;
				verifyAccuracy(editor.selection.active.line, jsonText.trim(), extensionContext);

				const edit = new vscode.WorkspaceEdit();
				edit.insert(filePath, editor.selection.active, jsonText);
				workspace.applyEdit(edit);

				// editor.edit(editBuilder => {
				// 	editBuilder.insert(editor.selection.active, jsonText);
				// });
			}
		});

		const provider2 = vscode.languages.registerCompletionItemProvider(
			'python',
			{
				provideCompletionItems(document: vscode.TextDocument, position: vscode.Position) {
					return [
						new vscode.CompletionItem('true', vscode.CompletionItemKind.Method),
						new vscode.CompletionItem('warn', vscode.CompletionItemKind.Method),
						new vscode.CompletionItem('error', vscode.CompletionItemKind.Method),
					];
				}
			},
			';', ':'
		);

		extensionContext.subscriptions.push(provider2);
	});

	extensionContext.subscriptions.push(disposable);

	let someTrackingIdCounter = 0;
	const provider: vscode.InlineCompletionItemProvider = {
		provideInlineCompletionItems: async (document, position, context, token) => {
			console.log('provideInlineCompletionItems triggered');

			const regexp = /\/\/ \[(.+),(.+)\):(.*)/;
			if (position.line <= 0) {
				return;
			}

			const documentText = document.getText();

			const response = await fetch("https://codefill-plugin.mikaturk.nl:8443/v1/autocomplete", {
				method: 'POST',
				body: JSON.stringify([documentText, ""]),
				headers: {
					'Content-Type': 'application/json',
					'Authorization': 'Basic ' + extensionContext.globalState.get('codefill-identifier')
				}
			});

			const json: any = await response.json();
			const jsonText = json.completion;
			// verifyAccuracy(editor.selection.active.line, jsonText.trim(), extensionContext);

			const lineBefore = document.lineAt(position.line - 1).text;
			console.log(lineBefore);

			const insertText = jsonText;

			return [
				{
					insertText,
					range: new vscode.Range(position.line, 4, position.line, editor!.document.lineAt(position.line).range.end.character),
					someTrackingId: someTrackingIdCounter++,
				},
			] as MyInlineCompletionItem[];

		},
	};

	vscode.languages.registerInlineCompletionItemProvider({ pattern: '**' }, provider);
}

interface MyInlineCompletionItem extends vscode.InlineCompletionItem {
	someTrackingId: number;
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