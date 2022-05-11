/* --------------------------------------------------------------------------------------------
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See License.txt in the project root for license information.
 * ------------------------------------------------------------------------------------------ */

import * as vscode from 'vscode';
import { workspace, ExtensionContext } from 'vscode';

import fetch from 'node-fetch';

export function activate(context: ExtensionContext) {

	console.log("test");
	
	const disposable = vscode.commands.registerCommand('codefill-plugin.sayHello', async () => {
		// The code you place here will be executed every time your command is executed

		const autocomplete_text = "def count_words(words):\n    ";

		const response = await fetch("https://codefill-plugin.mikaturk.nl:8443/v1/autocomplete", {
			method: 'POST', 
			body: JSON.stringify([autocomplete_text,""]), 
			headers: {
				'Content-Type': 'application/json'
			}
		});
		const json: any = await response.json();
		vscode.window.showInformationMessage(json.completion);

		// Display a message box to the user
		// vscode.window.showInformationMessage('Welcome to CodeFill IDE Auto Complete! :)');
	});

	context.subscriptions.push(disposable);
}

// eslint-disable-next-line @typescript-eslint/no-empty-function
export function deactivate() {}
