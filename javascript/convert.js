import jsTokens from "js-tokens";

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
            case "MultiLineComment":
            case "SingleLineComment":
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

            case "WhiteSpace": break;
            case "LineTerminatorSequence": {
                // Remove empty lines
                // TODO: account for lines with only whitespace
                if (prevToken.type == "LineTerminatorSequence") break;
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
