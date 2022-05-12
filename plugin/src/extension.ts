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

export function activate(context: ExtensionContext) {

	if (!context.globalState.get('codefill-identifier')) {
		context.globalState.update('codefill-identifier', rand(128, 32));
	}

	const disposable = vscode.commands.registerCommand('codefill-plugin.generateAutoComplete', async () => {

		const editor = vscode.window.activeTextEditor;

		vscode.window.withProgress({
			location: vscode.ProgressLocation.Notification,
			title: "Generating code completion... Please wait.",
			cancellable: true
		}, async (progress, token) => {
			token.onCancellationRequested(() => {
				console.log("Cancelled code completion...");
			});

			if (editor) {
				const document = editor.document;
				const documentText = document.getText();

				const response = await fetch("https://codefill-plugin.mikaturk.nl:8443/v1/autocomplete", {
					method: 'POST',
					body: JSON.stringify([documentText, ""]),
					headers: {
						'Content-Type': 'application/json',
						'Authorization': 'Basic ' + context.globalState.get('codefill-identifier')
					}
				});

				const json: any = await response.json();

				await editor.edit(editBuilder => {
					editBuilder.insert(editor.selection.active, json.completion);
				});
			}
		});
	});

	context.subscriptions.push(disposable);
}

// eslint-disable-next-line @typescript-eslint/no-empty-function
export function deactivate() { }