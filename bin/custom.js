require(["codemirror/keymap/sublime", "notebook/js/cell", "base/js/namespace"],
    function(sublime_keymap, cell, IPython) {
        // setTimeout(function(){ // uncomment line to fake race-condition
        cell.Cell.options_default.cm_config.keyMap = 'sublime';

        // not working:
        // cell.Cell.options_default.cm_config.extraKeys[‘Cmd-Enter’] = function(){console.log(‘cmd-enter’)};
		// cell.Cell.options_default.cm_config.extraKeys[‘Ctrl-Enter’] = function(){console.log(‘ctrl-enter’)};
		// cell.Cell.options_default.cm_config.extraKeys[‘Shift-Enter’] = function(){};

        var cells = IPython.notebook.get_cells();

        for(var cl=0; cl< cells.length ; cl++){
			cells[cl].code_mirror.setOption('extraKeys',
			{
				// for more - check: notebook/static/components/codemirror/keymap/sublime.js
				"Shift-Ctrl-Up": "addCursorToPrevLine",
				"Shift-Ctrl-Down": "addCursorToNextLine",
				"Cmd-Enter": function(){},
				"Ctrl-Enter": function(){}
			});
			cells[cl].code_mirror.setOption('keyMap', 'sublime');
		};

		console.log("@@@@@@@@ custom.js loaded!")


        // }, 1000)// uncomment  line to fake race condition
    }
);