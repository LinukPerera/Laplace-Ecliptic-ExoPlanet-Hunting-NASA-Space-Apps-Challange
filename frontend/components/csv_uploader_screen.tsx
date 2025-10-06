import { DotLottieReact } from '@lottiefiles/dotlottie-react';
import { useCallback, useState } from 'react';
import { FiUpload } from "react-icons/fi";
import predictFromModel from '@/lib/predict_from_model';
import ProgressBar from './progress_bar';
import Link from 'next/link';

function StateIdle({handleFileChange}:{handleFileChange: (event: React.ChangeEvent<HTMLInputElement>) => void}){
    return (
        <>
            <h2 className="text-black text-lg font-semibold mb-2 text-center">Download sample csv file and add the input data according to the format.</h2>
            <Link href="/KOISample.csv" download className="mb-6 px-4 py-0 text-purple-500 rounded hover:text-purple-600 transition-colors">Download Sample CSV</Link>
            <div className="csv-uploader p-4 w-full flex flex-col items-center relative min-h-48 border-2 border-dashed border-purple-300 rounded-lg cursor-pointer hover:border-purple-500 transition-colors hover:bg-purple-100">
                <FiUpload className="text-accent text-6xl mb-4" />
                {/*<DotLottieReact src="/anims/Rocket Loader.lottie" loop autoplay className="w-24 h-24 mb-4" /> */}
                <input type="file" accept=".csv" className="file-input absolute inset-0 opacity-0" onChange={handleFileChange} />
                <h1 className="text-2xl font-bold mb-4 text-center text-gray-700">Upload CSV File</h1>
                <p className="text-center text-gray-600">Please upload a CSV file to proceed with the prediction.</p>
            </div>
        </>
    )
}

function StateFileSelected({fileInfo, onClickProceed}:{fileInfo:{name:string; size:number}, onClickProceed: ()=>void}){
    return (
        <>
            <h1 className="text-black text-xl font-semibold mb-4">File Selected</h1>
            <div className="file-info p-4 w-full flex flex-col items-center relative min-h-48 border-2 border-dashed border-gray-300 rounded-lg cursor-pointer hover:border-gray-500 transition-colors">
                <h1 className="text-2xl font-bold mb-4 text-center text-gray-700">File: {fileInfo.name}</h1>
                <p className="text-center text-gray-600">Size: {(fileInfo.size / 1024).toFixed(2)} KB</p>
                <button className="mt-4 px-4 py-2 bg-indigo-500 text-white rounded hover:bg-indigo-800 transition-colors"
                    onClick={onClickProceed}
                    >Process File</button>
            </div>
        </>
    )
}

function StateFileUploading() {
    return (
        <>
            <h1 className="text-black text-xl font-semibold mb-4">Uploading & Processing</h1>
            <div className="file-info p-4 w-full flex flex-col items-center relative min-h-48 border-2 border-dashed border-gray-300 rounded-lg cursor-pointer hover:border-gray-500 transition-colors">
                <DotLottieReact src="/anims/Rocket Loader.lottie" loop autoplay className="w-28 h-28 mb-4" /> 
                <h1 className="text-2xl font-bold mb-4 text-center text-gray-700">Processing</h1>
            </div>
        </>
    )
}




interface ModelResult {
    keid: string;
    kepler_name: string;
    result: string;
}

function ResultColorBadge({result}:{result:string}) {
    let colorClass = "";
    if (result === "CONFIRMED") colorClass = "bg-emerald-100 text-emerald-800 border-emerald-300";
    else if (result === "FALSE POSITIVE") colorClass = "bg-rose-100 text-rose-800 border-rose-300";
    else colorClass = "bg-yellow-100 text-yellow-800 border-yellow-300";
    return (
        <div className={`w-fit text-sm font-medium mr-2 px-2.5 py-0.5 rounded border ${colorClass}`}>
            {result}
        </div>
    );
}


function StateCompleted({results, downloadResultsAsCSV, resetFn}:{results: ModelResult[], downloadResultsAsCSV: ()=>void, resetFn: ()=>void}) {

    // remove empth kepler_name and keid entries
    const filteredResults = results.filter((res, index, self) => res.kepler_name && res.keid && res.kepler_name.trim() !== "" && res.keid.trim() !== "");


    return (
        <>
        <h1 className="text-black text-xl font-semibold mb-4">Processing Completed</h1>
        <table className="min-w-full border-separate border-spacing-0 overflow-hidden ">
            <thead className="bg-gray-100">
                <tr>
                <th className="px-6 py-3 text-left text-sm font-semibold text-gray-600 border-b">Kepler ID</th>
                <th className="px-6 py-3 text-left text-sm font-semibold text-gray-600 border-b">Kepler Name</th>
                <th className="px-6 py-3 text-left text-sm font-semibold text-gray-600 border-b">Prediction Result</th>
                </tr>
            </thead>
            <tbody>
                {filteredResults.map((res, index) => (
                <tr
                    key={index}
                    className="hover:bg-gray-50 transition-colors duration-150 even:bg-gray-50/50"
                >
                    <td className="px-6 py-3 text-sm text-gray-700 border-b">{res.keid}</td>
                    <td className="px-6 py-3 text-sm text-gray-700 border-b">{res.kepler_name}</td>
                    <td className="px-6 py-3 text-sm text-gray-700 border-b">
                        <ResultColorBadge result={res.result} />
                    </td>
                </tr>
                ))}
            </tbody>
        </table>
        <div className='flex items-start flex-col gap-2 text-sm font-mono p-4 border border-gray-200 rounded mb-4 w-full bg-amber-50 mt-4'>
            <div className="w-full flex items-center">
                <div className='bg-emerald-100 text-emerald-800 text-sm font-medium mr-2 px-2.5 py-0.5 rounded border border-emerald-300'>CONFIRMED</div>
                <div>High confidence of having exoplanent planet.</div>
            </div>
            <div className="w-full flex items-center">
                <div className='bg-yellow-100 text-yellow-800 text-sm font-medium mr-2 px-2.5 py-0.5 rounded border border-yellow-300'>CANDIDATE</div>
                <div>Further observation needed to confirm exoplanet.</div>
            </div>

            <div className="w-full flex items-center">
                <div className='bg-rose-100 text-rose-800 text-sm font-medium mr-2 px-2.5 py-0.5 rounded border border-rose-300'>FALSE POSITIVE</div>
                <div>No evidence of exoplanet detected.</div>
            </div>
        </div>
        <div className="flex items-center justify-center">
            <button className='mt-4 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 transition-colors' onClick={downloadResultsAsCSV}>Download CSV</button>
            <button className='mt-4 px-4 py-2 bg-gray-500 text-white rounded hover:bg-gray-600 transition-colors ml-4' onClick={resetFn}>Process Another File</button>
        </div>
        </>
    );
}

function StateError({resetFn}:{resetFn: ()=>void}) {
    return (
        <>
        <DotLottieReact src="/anims/Planet Orbit.lottie" loop autoplay className="w-36 h-36 mb-4" />
        <div className="text-red-600 font-semibold mb-4">An error occurred during processing. Please try again.</div>

        <div className="flex items-center justify-center">
            <button className='mt-4 px-4 py-2 bg-gray-500 text-white rounded hover:bg-gray-600 transition-colors ml-4' onClick={resetFn}>Try Again</button>
        </div>
        </>
    );
}
// for testing purpose
/*const sampleModelResults: ModelResult[] = [
    { keid: "123456", kepler_name: "Kepler-22b", result: "CONFIRMED" },
    { keid: "789012", kepler_name: "Kepler-69c", result: "FALSE POSITIVE" },
    { keid: "345678", kepler_name: "Kepler-62f", result: "CANDIDATE" },
    { keid: "123456", kepler_name: "Kepler-22b", result: "CONFIRMED" },
    { keid: "789012", kepler_name: "Kepler-69c", result: "FALSE POSITIVE" },
    { keid: "345678", kepler_name: "Kepler-62f", result: "CANDIDATE" },
    { keid: "123456", kepler_name: "Kepler-22b", result: "CONFIRMED" },
    { keid: "789012", kepler_name: "Kepler-69c", result: "FALSE POSITIVE" },
    { keid: "345678", kepler_name: "Kepler-62f", result: "CANDIDATE" },
    { keid: "789012", kepler_name: "Kepler-69c", result: "FALSE POSITIVE" },
    { keid: "345678", kepler_name: "Kepler-62f", result: "CANDIDATE" },
    { keid: "123456", kepler_name: "Kepler-22b", result: "CONFIRMED" },
    { keid: "789012", kepler_name: "Kepler-69c", result: "FALSE POSITIVE" },
    { keid: "345678", kepler_name: "Kepler-62f", result: "CANDIDATE" }
];*/


function downloadCSV(results: ModelResult[]) {
    const headers = ["keid", "kepler_name", "result"];
    const csvRows = [
        headers.join(","), // Header row
        ...results.map(res => [res.keid, res.kepler_name, res.result].join(","))
    ];
    const csvContent = csvRows.join("\n");
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'model_results.csv';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}


export default function CSVFileUploader() {
    const [uploaderState, setUploaderState] = useState<'idle' | 'file-selected' | 'processing' | 'completed'>('idle');
    const [selectedFile, setSelectedFile] = useState<File | null>(null);
    const [selectedFileInfo, setSelectedFileInfo] = useState<{ name: string; size: number } | null>(null);
    const [modelResults, setModelResults] = useState<ModelResult[] | null>(null);

    const handleFileChange = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0] || null;
        setSelectedFile(file);
        if (file) {
            const fileInfo = { name: file.name, size: file.size };
            setSelectedFileInfo(fileInfo);
            setUploaderState('file-selected');
        }
    }, []);

    const handleFileUploadAndProcessBtnClick = useCallback(async ()=>{
        if (!selectedFile) return;

        setUploaderState("processing");

        const fileText = await selectedFile.text();

        predictFromModel(fileText).then((results) => {
            setModelResults(results);
            setUploaderState("completed");
        }).catch((error) => {
            console.error("Error during prediction:", error);
            setUploaderState("completed");
        });

        // setTimeout(() => {
        //     setModelResults(sampleModelResults);
        //     setUploaderState("completed");
        // }, 3000); // Simulate a 3-second processing time

    }, [selectedFile]);

    const doRest = useCallback(() => {
        setUploaderState("idle");
        setSelectedFile(null);
        setSelectedFileInfo(null);
        setModelResults(null);
    }, []);


    const downloadResultsAsCSV = useCallback(() => {
        if (modelResults) {
            downloadCSV(modelResults);
        }
    }, [modelResults]);

    return (
        <div className="flex flex-col items-center w-full p-4">
            <ProgressBar step={uploaderState} />
            {uploaderState === 'idle' && <StateIdle handleFileChange={handleFileChange} />}
            {uploaderState === 'file-selected' && selectedFileInfo && <StateFileSelected fileInfo={selectedFileInfo} onClickProceed={handleFileUploadAndProcessBtnClick} />}
            {uploaderState === 'processing' && <StateFileUploading /> }
            {uploaderState === 'completed' && modelResults && <StateCompleted results={modelResults} downloadResultsAsCSV={downloadResultsAsCSV} resetFn={doRest} />}
            {uploaderState === 'completed' && !modelResults && <StateError resetFn={doRest} />}
        </div>
    );
}