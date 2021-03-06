import { readFile, writeFile } from "node:fs/promises"
import jsTokens from "js-tokens";

/**
 * Converts a JavaScript file into a token file for use in CodeFill.
 * @param {string} str A JavaScript file represented by a string
 * @returns {string} The token form of the input file
 */
export function convert(str) {
    const tokens = jsTokens(str);
    const output_parts = [];
    let output_line = [];
    let prevToken = null;

    for (const token of tokens) {
        switch (token.type) {
            case "StringLiteral":
            case "NoSubstitutionTemplate":
            case "TemplateHead":
            case "TemplateMiddle":
            case "TemplateTail":
            case "RegularExpressionLiteral":
            case "IdentifierName":
            case "PrivateIdentifier":
            case "NumericLiteral":
            case "Punctuator":
            case "Invalid": {
                output_line.push(token.type);
                // For debugging
                // output_line.push(token.type + `\`${token.value}\``);
                break;
            }

            case "MultiLineComment":
            case "SingleLineComment":
            case "WhiteSpace": break;

            case "LineTerminatorSequence": {
                // Only add a line if it has content
                if (output_line.length > 0)
                    output_parts.push(output_line.join(' '));
                output_line = [];
                break;
            }
        }

        prevToken = token;
    }
    output_parts.push(output_line.join(' '));

    return output_parts.join('\n');
}

/**
 * Converts a JavaScript file into a token file for use in CodeFill.
 * @param {string} path Path to input JavaScript file
 * @param {string} output_path Path to output token file
 * @returns
 */
export async function convert_file(path, output_path) {
    const source_str = await readFile(path).then(x=>x.toString())
    const converted = convert(source_str);
    await writeFile(output_path, converted);
}

if (process.argv.length == 4) {
    // Example: 
    // $ node convert.js file.js output.txt
    convert_file(process.argv[2], process.argv[3])
} else if (process.argv.length == 4) {
    // For debugging purposes only, using console.log as output
    
    // Example: 
    // $ node convert.js file.js
    convert_file(process.argv[2])
}