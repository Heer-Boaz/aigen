from contextlib import asynccontextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from PIL import Image
from textual import events
from textual.widgets import Button, Input, Label, OptionList, TabbedContent

from aigen import image_tui
from aigen.tui_choice_menu import ChoiceMenu
from aigen.tui_dialogs import ConfirmationDialog, MessageDialog, PromptDialog
from aigen.tui_file_browser import FileBrowser
from aigen.tui_form_fields import FieldRow
from aigen.tui_text_editor import MultilineInput
from aigen.sam_prompt_dialog import SAMPromptDialog
from aigen.workflow_document_io import load_workflow_document
from aigen.workflow_form_import import image_form_workflow, video_form_workflow
from aigen.workflow_graph import ImageEditNode
from aigen.workflow_property_widgets import PropertyTextArea
from test_workflow_interactions import interaction_graph
from test_workflow_properties import open_editor, property_widget, select_node


@asynccontextmanager
async def open_app(directory, size=(80, 24)):
    with patch.object(image_tui, 'STATE_PATH', directory / 'form.json'):
        app = image_tui.ImageGenerationApp()
        async with app.run_test(size=size) as pilot:
            await pilot.pause()
            yield app, pilot


def prompt_row(app, form):
    return next(row for row in app.query(FieldRow) if row.form is form and row.field.name == 'prompt')


class TUIEditingTests(unittest.IsolatedAsyncioTestCase):
    async def test_forms_keep_multiline_text_and_undo_through_refresh_resize_and_import(self):
        value = '\nFirst line\n\n  Second line – café 日本語  \n'
        with TemporaryDirectory() as directory:
            async with open_app(Path(directory)) as (app, pilot):
                for tab, form, convert in (('images', app.form, image_form_workflow),
                                           ('videos', app.video_form, video_form_workflow)):
                    app.query_one(TabbedContent).active = tab
                    await pilot.pause()
                    row = prompt_row(app, form)
                    text = row.query_one(MultilineInput)
                    text.focus()
                    app.post_message(events.Paste(value))
                    await pilot.pause()
                    self.assertEqual(text.text, value)
                    self.assertEqual(form.field('prompt').value, value)
                    await app._rebuild_form(form)
                    await pilot.resize_terminal(120, 40)
                    self.assertIs(prompt_row(app, form).query_one(MultilineInput), text)
                    await pilot.press('ctrl+z')
                    self.assertEqual(form.field('prompt').value, '')
                    await pilot.press('ctrl+y')
                    self.assertEqual(form.field('prompt').value, value)
                    graph = convert(form)
                    self.assertTrue(any(getattr(node.config, 'prompt', None) == value for node in graph.nodes))
                    await pilot.resize_terminal(80, 24)

    async def test_editing_keys_never_remove_a_hovered_input(self):
        with TemporaryDirectory() as directory:
            async with open_app(Path(directory)) as (app, pilot):
                slot = next(row for row in app.query(FieldRow) if row.form is app.form and row.field.slot_kind == 'image')
                text = prompt_row(app, app.form).query_one(MultilineInput)
                text.focus()
                app.post_message(events.Paste('abc'))
                await pilot.pause()
                slot.add_class('hovered')
                before = tuple(app.form.fields)
                await pilot.press('backspace')
                self.assertEqual(text.text, 'ab')
                self.assertEqual(tuple(app.form.fields), before)

    async def test_footer_labels_and_new_workflow_menu_are_readable_at_80_columns(self):
        with TemporaryDirectory() as directory:
            async with open_app(Path(directory)) as (app, pilot):
                for tab in ('images', 'videos', 'sam-edit', 'postprocessing', 'workflows'):
                    app.query_one(TabbedContent).active = tab
                    await pilot.pause()
                    for button in app.query('#action-footer Button'):
                        if button.is_on_screen:
                            self.assertGreaterEqual(button.content_region.width, len(str(button.label)))
                            self.assertLessEqual(button.region.right, 80)
                button = next(button for button in app.query(Button) if button.name == 'workflow-new-menu')
                await pilot.click(button)
                self.assertIsInstance(app.screen, ChoiceMenu)
                options = app.screen.query_one(OptionList)
                self.assertEqual(options.option_count, 3)
                self.assertIn('Character workflow', str(options.get_option_at_index(1).prompt))
                await pilot.press('escape')
                self.assertIs(app.focused, button)

    async def test_escape_returns_from_dialogs_and_cancels_confirmation(self):
        with TemporaryDirectory() as directory:
            directory = Path(directory)
            image = directory / 'input.png'
            Image.new('RGB', (8, 8), 'white').save(image)
            async with open_app(directory) as (app, pilot):
                opener = prompt_row(app, app.form).query_one(MultilineInput)
                callbacks = []
                for dialog in (MessageDialog('Message', 'Details'), PromptDialog('Name', 'Name'),
                               ConfirmationDialog('Discard', 'Discard edits?', confirm_label='Discard'),
                               FileBrowser(directory, title='Open', directories_only=False,
                                           extensions=frozenset({'.png'}), select_label='Open'),
                               SAMPromptDialog(image=str(image), prompt_mode='box', box='',
                                               positive_points='', negative_points='')):
                    opener.focus()
                    app.push_screen(dialog, callbacks.append)
                    await pilot.pause()
                    await pilot.press('escape')
                    await pilot.pause()
                    self.assertIsNot(app.screen, dialog)
                    self.assertIs(app.focused, opener)
                self.assertEqual(callbacks, [None, None, False, None, None])

    async def test_save_workflow_selects_directory_and_filename_in_one_dialog(self):
        with TemporaryDirectory() as directory:
            directory = Path(directory)
            async with open_app(directory) as (app, pilot):
                with patch.object(image_tui, 'PROJECT_ROOT', directory):
                    app._choose_workflow_directory()
                await pilot.pause()
                browser = app.screen
                self.assertIsInstance(browser, FileBrowser)
                browser.query_one('#browser-save-name', Input).value = '../outside.json'
                await pilot.click('#browser-select')
                self.assertIs(app.screen, browser)
                browser.query_one('#browser-save-name', Input).value = 'saved-flow'
                browser.query_one('#browser-save-name').focus()
                await pilot.press('enter')
                await pilot.pause()
                self.assertEqual(app.workflow_buffer.document_path, directory / 'saved-flow.json')
                self.assertEqual(load_workflow_document(directory / 'saved-flow.json'), app.workflow_buffer.document)
