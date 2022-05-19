import * as vscode from 'vscode';
import fetch from 'node-fetch';

var isActive = false;

export class CodeCompletionProvider implements vscode.CompletionItemProvider {
    extensionContext: vscode.ExtensionContext;
    constructor(extensionContext: vscode.ExtensionContext) {
        this.extensionContext = extensionContext;
    }
    provideCompletionItems(document: vscode.TextDocument, position: vscode.Position, token: vscode.CancellationToken, context: vscode.CompletionContext): vscode.ProviderResult<vscode.CompletionList<vscode.CompletionItem> | vscode.CompletionItem[]> {
        // const [nativeCompletionList, val] = await Promise.all([
        // 	vscode.commands.executeCommand<vscode.CompletionList>('vscode.executeCompletionItemProvider', document.uri, position),
        // 	callToAPIAndRetrieve(document, extensionContext)
        // ]);
        // console.log("val", val);

        // if (!val) return undefined;
        return createCompletionList(document, position, this.extensionContext);
    }
}

async function createCompletionList(document: vscode.TextDocument, position: vscode.Position, extensionContext: vscode.ExtensionContext) {
    let completionList: vscode.CompletionList<vscode.CompletionItem> = new vscode.CompletionList();
    
    if (isActive) return [];
    isActive = true;
    const val = await callToAPIAndRetrieve(document, extensionContext);
    if (!val) return [];
    // const nativeCompletionList = await vscode.commands.executeCommand<vscode.CompletionList>('vscode.executeCompletionItemProvider', document.uri, position);
    // const val = "test";
    // console.log(nativeCompletionList);
    const simpleCompletion = new vscode.CompletionItem(val, vscode.CompletionItemKind.Method);
    simpleCompletion.sortText = "0.0.", simpleCompletion.label;
    completionList.items.push(simpleCompletion);
    // completionList.items.push(...nativeCompletionList.items)
    console.log(completionList);
    isActive = false;
    return completionList
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