import { readdir, readFile, writeFile, mkdir } from "node:fs/promises"
import { join } from "node:path"
import { convert } from "./convert.js";

async function convert_file(path, output_path) {
    const source_str = await readFile(path).then(x=>x.toString())
    const converted = convert(source_str);
    await writeFile(output_path, converted);
}

/**
 * Convert a whole directory of JavaScript source files to token representation.
 * @param {string} source_dir The directory where the source JavaScript files are located.
 * @param {string} converted_dir The directory where the converted files should be placed.
 * @returns {void}
 */
export async function convert_dir(source_dir, converted_dir) {
    const files = await readdir(source_dir);

    await mkdir(converted_dir, { recursive: true })
    // Single threaded loop
    // TODO: Add multithreading
    const convert_promises = files.map(file => {
        const path = join(source_dir, file);
        const converted_path = join(converted_dir, file + ".txt");
        return convert_file(path, converted_path);
    });

    await Promise.all(convert_promises);
}

const JS_FOLDER = "/mnt/mturk/cf_sample_data/javascript/"
const JS_SRC_FOLDER = join(JS_FOLDER, "js-dataset")
const JS_CONVERTED_FOLDER = join(JS_FOLDER, "converted")

if (process.argv.length == 4) {
    convert_dir(process.argv[2], process.argv[3])
} else {
    convert_dir(JS_SRC_FOLDER, JS_CONVERTED_FOLDER);
}
